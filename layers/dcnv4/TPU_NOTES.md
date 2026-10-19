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

## 进展日志

- 2026-10-19：
  - 创建 branch `dcnv4-tpu`
  - 摸清 TPU 连接方式（TFRC，grpc://localhost:8470 由 tpu-runtime 容器提供；resolver 用 TPU_NAME=node4）
  - 踩坑：客户端 jit_compile 会把服务端搞崩 → 确认必须走 strategy.run 路线
  - 下一步：写 `benchmark_dcnv4_tpu.py`（TPUStrategy + tf.data + strategy.run，无客户端 jit_compile），先测 DCNv2 vs DCNv4 基线
