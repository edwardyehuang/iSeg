# DCNv4 TPU 优化 — 上下文笔记

> 这个文件记录任务上下文与进展，供会话恢复时使用。每次有重要进展都要更新并提交。

## 任务

- 目标：让 DCNv4（`iseg/layers/dcnv4`）在 TPU（v4-8）上跑得**比 DCNv2（XLA）快得多**。
- 改进代码放在 `iseg/layers/dcnv4/dcnv4_tpu.py`。
- 不能用自定义 op，只能用现有 TF op + XLA。
- 对比基准：`iseg/layers/dcn_v2.py` 的 DCNv2（XLA）。
- GPU XLA 已适配（`dcnv4.py`），但其加速优势在 TPU 上没体现出来。
- git branch：`dcnv4-tpu`（已创建并 push）。**每跑一轮 TPU 就 commit + push + 更新本文件。**

## 环境

- TPU VM：`tpu-vm-tf-2.15.0-pod-se`，TPU v4-8（4 chips / 8 cores），us-central2-b
- 本机 TF 2.15.0（pip），keras3 兼容靠 `legacy_keras_patch.apply_patch`
- 环境变量：`TPU_NAME=node4`，`TPU_LOAD_LIBRARY=0`（**必须保持 0**，直连模式会与 tpu-runtime 容器抢设备并搞崩 runtime）
- TPU runtime 跑在 docker 容器 `tpu-runtime` 里，grpc 监听 `localhost:8470`
- **每次用完 TPU 后必须 `sudo systemctl restart tpu-runtime.service`**，否则下次可能卡住/连上已污染实例

## 怎么用 TPU（关键，已验证）

- 代码库入口模式：`TPUClusterResolver(tpu_name)` → `experimental_connect_to_cluster` → `tf.tpu.experimental.initialize_tpu_system` → `tf.distribute.TPUStrategy`（见 `iseg/distribution/distribution_utils.py:get_tpu_strategy`）
- 生产启动方式（不要跑这个 cfg）：`python3 main.py --flagfile=... --tpu_name=${TPU_NAME} --use_tpu_pod=True`
- **XLA 的正确打开方式**：客户端上**不要**用 `tf.function(jit_compile=True)`。TPU 上的 XLA 编译发生在服务端（tpu-runtime 容器），经由 `strategy.run` → `tpu.replicate` 自动完成。keras fit 内部就是这条路。
- **坑（已踩过）**：客户端 `jit_compile=True` + host 输入会触发服务端 "could not find registered transfer manager for platform Host" 错误，会 **poison 服务端 EagerExecutor 并最终让 runtime 崩溃**（Fatal Python error: Aborted → segfault，systemd 自动重启）。表现是客户端报 Socket closed / "worker was restarted"。
- TPU 训练用 `mixed_bfloat16`（`iseg/utils/common.py:enable_mixed_precision(use_tpu=True)`）。

## 现状分析

- `dcnv4.py`：纯 TF op 实现，4 次 gather（每角点一次 `[B,Ho,Wo,G,K,D]`），group 结构 D=C/G。
- TPU 上可能的性能杀手（待实测验证）：
  - gather 的行宽只有 D=C/G（如 8~32），远低于 TPU 128-lane 向量宽度
  - 4 次独立 gather + 6D tensor 的 reduce
  - bf16 下 gather 带宽减半但仍受限于行宽

## 基线实测（bf16 mixed precision，B=4=每replica 1个）

Config 1：B=4, 128×128, C=128, G=4
- dcnv2: fwd=5.62ms, fwd+bwd=6.82ms（backward 几乎免费！）
- dcnv4: fwd=9.14ms, fwd+bwd=64.42ms（backward 55ms，灾难）
- dcnv4 vs dcnv2: fwd 0.61x，fwd+bwd 0.11x

Config 2：B=4, 256×256, C=256, G=4
- dcnv2: fwd=140.06ms, fwd+bwd=139.49ms
- dcnv4: fwd=138.62ms, fwd+bwd=259.71ms
- dcnv4 vs dcnv2: fwd 1.01x，fwd+bwd 0.54x

关键结论：
- 大尺寸下两者 fwd 打平且都被 gather 卡死：2.4G 元素 gather 实测 ~140ms → 有效带宽仅 ~9GB/s/chip（TPU v4 HBM 标称 ~1.2TB/s），说明 XLA:TPU 的 gather lowering 是 per-slice 开销主导的，极慢
- dcnv2 的 backward 真免费（CPU 验证过 grads 非空非零），dcnv4 的 backward（4 次 scatter + 权重反传）慢得离谱
- **方向确定：彻底消除 per-position 动态 gather/scatter，改走 dense window + einsum 路线**（im2col 式静态切片 + 动态 dense 权重 + matmul，全是 TPU 原生快路径）

## 消融实测（core 级，B=4, 128×128, C=128, fp32 输入）

| 变体 | fwd | fwd+bwd | 结论 |
|---|---|---|---|
| full（原版） | 9.80 | 63.87 | 基线 |
| const_idx（常量索引gather） | 6.22 | 56.08 | 索引数学 fwd ~3.6ms；bwd 仍慢 |
| static_slice（无gather） | 6.05 | **7.15** | **去掉 gather 后 bwd 几乎免费** |
| fused_gather（4合1大gather） | 15.70 | 80.27 | 大 gather 更差 |

→ backward 灾难 = gather 的反向 scatter；fwd 也受 gather 拖累且随尺寸超线性恶化（128² ~3.6ms → 256² ~120ms）

## dcnv4_tpu.py 迭代记录

- v1（einsum 'bhwgkt,bhwtgd->bhwgd' 批量小 dot）：**失败** — fwd=63.6ms @128²（XLA:TPU 对百万级 per-position 小 dot 的 lowering 极差）。但 fwd+bwd≈fwd（72ms），验证 dense 后 backward 免费的判断。
- v2：**K 先坍缩进权重**（数学上精确：out = Σ_t U[t]·Σ_k w[k,t]，因为 k 只出现在权重里）→ 权重 [B,H,W,G,T2]，聚合用 broadcast-mul + reduce_sum。CPU parity 全过（窗口内 exact ~1e-14）。
- v3（extract_patches 换 slices+stack）：**突破** — extract_patches 在 XLA:TPU 上 +59ms @128²；T² 次 slice+stack 同样语义几乎免费。fwd 1.55ms @128²（3.58x vs dcnv2）。
- v4（one-hot depthwise conv 做窗口提取）：**失败** — multiplier=T² 的 depthwise 在 TPU 上是慢路径，fwd=75ms @128²。
- v5（当前最佳）：slices+stack 前向 + **tf.custom_gradient 手写 value adjoint**（64 项 mul→pad→add_n，替代 autodiff 的 slice-backward）。@128²: fwd 1.47ms / fwd+bwd 4.15ms（4.15x / 1.69x vs dcnv2）；@256²C=256: fwd 12.5ms / fwd+bwd 31.7ms（2.70x / 1.36x）。

## 重要负面结果 / 死路

- 客户端 `tf.function(jit_compile=True)`：把 tpu-runtime 服务端搞崩（Host transfer manager 缺失 → poison → segfault）
- tf.data 逐步喂数据做 benchmark：infeed 瓶颈（134MB/step → 140ms floor），必须 on-device 循环依赖输入
- `* 0.0` 强制计算梯度：被 XLA DCE 掉（用 `* 1e-30`）
- tf.image.extract_patches：+59ms @128²（XLA:TPU lowering 灾难）
- 批量小 dot einsum（百万级 batch）：63ms @128²
- one-hot depthwise conv（multiplier=64）：75ms fwd @128²
- separable 2-pass 分解：数学上不成立（per-position 权重挂在输出位置，1D 分解必然错位）
- 反向 col2im 的 conv/matmul 化：one-hot filter 的 regular conv2d_backprop_input 有 C× 冗余 → 146 TFLOP，死路；depthwise backprop 同死路

## 当前瓶颈认知（256² C=256, bf16, B=1/replica）

- 全链路 XLA 实际带宽利用率 ~1/3（fwd 20GB→12.5ms ≈ 1.6TB/s 聚合）
- fwd+bwd 31.7ms 中反向 ~19ms：d_weights 读 patches（8.6GB）+ 手写 adjoint 64 项 pad/add（~17GB）+ 权重数学反向
- 剩余杠杆：缩窗口（R=2→R=1 流量 ×0.56）、batch 增大摊薄固定开销

## 进展日志

- 2026-10-19：
  - 创建 branch `dcnv4-tpu`
  - 摸清 TPU 连接方式（TFRC，grpc://localhost:8470 由 tpu-runtime 容器提供；resolver 用 TPU_NAME=node4）
  - 踩坑：客户端 jit_compile 会把服务端搞崩 → 确认必须走 strategy.run 路线
  - 写 `benchmark_dcnv4_tpu.py`（TPUStrategy + tf.data + strategy.run，无客户端 jit_compile）
  - 踩坑2：`*0.0` 强制作梯度会被 XLA DCE 掉 → 改 `*1e-30`；`tf.constant` 传参会 retrace → `input_signature` 固定一次 trace
  - 基线数据（见上）：dcnv4 fwd+bwd 比 dcnv2 慢 2~9 倍
  - 写 `profile_core_tpu.py` 消融脚本（定位 gather/index/weights 占比）
  - dense window 路线确立 → `dcnv4_tpu.py`：CPU parity（窗口内 f64 exact ~1e-14，bf16 ~0.7%）
  - extract_patches 定位 → slices+stack；infeed 瓶颈定位 → on-device benchmark
  - 自定义梯度修 backward；当前最佳 v5（见上）

## 进展日志

- 2026-10-19：
  - 创建 branch `dcnv4-tpu`
  - 摸清 TPU 连接方式（TFRC，grpc://localhost:8470 由 tpu-runtime 容器提供；resolver 用 TPU_NAME=node4）
  - 踩坑：客户端 jit_compile 会把服务端搞崩 → 确认必须走 strategy.run 路线
  - 写 `benchmark_dcnv4_tpu.py`（TPUStrategy + tf.data + strategy.run，无客户端 jit_compile）
  - 踩坑2：`*0.0` 强制作梯度会被 XLA DCE 掉 → 改 `*1e-30`；`tf.constant` 传参会 retrace → `input_signature` 固定一次 trace
  - 基线数据（见上）：dcnv4 fwd+bwd 比 dcnv2 慢 2~9 倍
  - 写 `profile_core_tpu.py` 消融脚本（定位 gather/index/weights 占比）
