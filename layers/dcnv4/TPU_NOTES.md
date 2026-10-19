# DCNv4/DCNv2 TPU 优化 — 上下文笔记

> 这个文件记录任务上下文与进展，供会话恢复时使用。每次有重要进展都要更新并提交。

## 任务与现状

- **DCNv4（已完成主线）**：`iseg/layers/dcnv4/dcnv4_tpu.py` 的 `DCNv4_TPU` / `dcnv4_core_tpu` 在 TPU 上比 DCNv2(XLA) 快得多（大分辨率 fwd 2.7-4.0x、fwd+bwd 1.5-3.5x）。已合并进 master：`dcnv4.py` 是统一自动路由层（TPU 上下文走窗口核心，否则走 gather 核心），`dcnv4_base.py` 是参考实现。
- **DCNv2（本次任务）**：master 已把 dcnv2 收进 `iseg/layers/dcnv2/`（dcnv2.py、dcnv2_flash.py）。目标：测 TPU 速度 + 开发 dcnv2 的 TPU 版本。
- 工作分支：`dcnv4-tpu`（iseg 子模块与外层 segmenty 都有同名分支，均已推送）。

## 环境（TPU v4-8，tpu-vm-tf-2.15.0-pod-se）

- 本机 TF 2.15.0（pip），keras3 兼容靠 `legacy_keras_patch.apply_patch`
- 环境变量：`TPU_NAME=node4`，`TPU_LOAD_LIBRARY=0`（**必须保持 0**；设 1 会与 tpu-runtime 容器抢 /dev/accel 并搞崩 runtime）
- TPU runtime 在 docker 容器 `tpu-runtime` 里，grpc 监听 `localhost:8470`（容器内 uid 2000）
- **每次用完 TPU（包括崩溃/异常退出的进程）后必须 `sudo systemctl restart tpu-runtime.service`**，否则下次连上已污染实例会 Socket closed / "worker was restarted"
- **每个用完 TPU 的 python 进程退出时可能挂住（TFRC 清理 hang）**——属正常，kill 即可
- 同一 runtime 实例不能被两个 python 进程顺序共用：第一个进程断连会污染它 → 每个 benchmark 进程用单独（重启后的）runtime

## 怎么用 TPU（已验证的正确姿势）

- 代码库入口：`get_tpu_strategy(tpu_name)`（`iseg/distribution/distribution_utils.py`）→ TPUStrategy，num_replicas=4（v4-8 暴露 4 个 device）
- **XLA 在服务端编译**（tpu.replicate 路径）：客户端**不要** `tf.function(jit_compile=True)`——会触发服务端 "no registered transfer manager for platform Host" 错误并**搞崩 runtime**
- TPU 训练用 `mixed_bfloat16`（`enable_mixed_precision(use_tpu=True)`）
- benchmark 必须 **on-device 循环依赖输入**（`x <- x + 1e-30*out`，防 XLA 的 LICM/CSE 把迭代折叠掉）：tf.data 逐步喂数会被 infeed 卡死（134MB/step → 140ms floor）；`*0.0` 强制作梯度会被 XLA DCE 掉（用 `*1e-30`）；`tf.constant` 传参会 retrace（用 `input_signature` 固定）
- profiler：`tf.profiler.experimental.client.trace("grpc://localhost:8470", logdir, ...)` 可抓 worker 的 xplane；logdir 要 chmod 777（容器 uid 2000 写共享 /tmp）；用 TF 自带 `tensorflow.core.profiler.protobuf.xplane_pb2` 解析（**不要** pip install tensorboard_plugin_profile——会拉 protobuf 7.x 打爆 TF 2.15；若已坏：`pip install "protobuf==4.25.3" "grpcio==1.54.3" "grpcio-status==1.54.3" --upgrade pyOpenSSL`）

## DCNv4-TPU 设计（已交付，关键结论）

dense window 替代 gather：双线性采样 = hat 函数在静态窗口上的加权和 → im2col（T² 静态 slice + stack）+ 权重外积 + mul+reduce。**窗口内与参考实现严格相等**（f64 ~1e-14，梯度同精确）；窗口外 soft-clamp（`R + tanh(|off|-R)`，梯度不断流）。

三档：`window_radius="exact"`（gather 参考，位级一致，慢）/ `R=2` / `R=1`。

反向全手写进 `tf.custom_gradient`：value adjoint = 64 项 (mul→pad)→add_n；权重路径闭式反向。否则 XLA 把跨 fwd/bwd 边界的长生命 buffer 卸到 host（S(3)，~10ms/step 实测）。

## DCNv4 最终成绩（bf16，B=1/replica；fwd / fwd+bwd，括号为 vs dcnv2 加速比）

| config | dcnv2 | dcnv4(原版) | dcnv4_tpu R=2 | dcnv4_tpu R=1 |
|---|---|---|---|---|
| 128² C=128 | 5.53 / 7.08 | 9.31 / 64.5 | 1.45 (3.8x) / 3.95 (1.79x) | 1.41 (3.9x) / 2.04 (3.47x) |
| 256² C=256 | 33.8 / 43.3 | 38.1 / 262.8 | 12.6 (2.7x) / 29.6 (1.46x) | 8.42 (4.0x) / 17.1 (2.54x) |
| 128² C=256 | 7.80 / 10.3 | 9.55 / 64.7 | 2.38 (3.3x) / 6.22 (1.66x) | 2.20 (3.6x) / 3.33 (3.11x) |
| 64² C=256 | 2.25 / 2.78 | 2.39 / 6.49 | 2.17 (1.0x) / 6.09 (0.46x) | 1.44 (1.6x) / 3.54 (0.78x) |
| 32² C=256 | 1.48 / 1.97 | 1.27 / 2.08 | 1.38 (1.1x) / 2.15 (0.92x) | 1.32 (1.1x) / 2.41 (0.82x) |
| 128² C=512 | 12.3 / 17.8 | 10.4 / 65.3 | 10.3 (1.2x) / 12.0 (1.48x) | 7.47 (1.7x) / 13.2 (1.34x) |
| 64² C=512 | 3.39 / 4.35 | 1.73 / 6.42 | 3.06 (1.1x) / 6.47 (0.67x) | 1.57 (2.2x) / 3.73 (1.16x) |
| B=8 128² C=256 | 18.4 / 22.7 | 19.5 / 130.0 | 5.97 (3.1x) / 15.4 (1.47x) | 4.74 (3.9x) / 7.75 (2.93x) |
| B=8 64² C=512 | 6.54 / 8.34 | 4.46 / 34.0 | 4.54 (1.4x) / 13.3 (0.62x) | 3.77 (1.7x) / 7.22 (1.16x) |

小分辨率（≤64²）fwd+bwd 是 launch-bound 弱项（64 项 adjoint 的 ~192 个 HLO op 发射开销）；大分辨率全胜。

## DCNv2-TPU 探索（本次）

### 基线（bf16，B=4；fwd / fwd+bwd ms）

| config | dcnv2 | dcnv2_flash |
|---|---|---|
| 128² C=256 | 8.06 / 10.64 | 3.80 / 8.05 |
| 256² C=256 | 33.81 / 42.99 | 17.60 / 40.65 |

### 核心障碍（负面结果）

DCNv2 的 **K 点必须保留**（最终 GEMM [K·C,F] 混合 K 与 C），不能像 DCNv4 坍缩。K-保留聚合 `out[k,c] = Σ_t W[k,t]·U[t,c]` 在 XLA:TPU 上**所有写法都是 per-position 标量循环**（@256² C=256：einsum 7213ms、mulreduce 7188ms、kloop 7055ms、kloop_t 4227ms、dcnv4style 4175ms；@64² 也 442ms——按位置线性缩放）。

**dcnv4 能快的本质**：W 在自由轴 D 上共享（per-g 向量×向量乘），XLA 可向量化；dcnv2 的 W、U 各有自由轴 K、C，是 per-position 小矩阵乘，不向量化。反向的 dU（K 求和 + (t,c) 自由）同样是 per-position 外积，也死。

### dcnv2_tpu 交付（`iseg/layers/dcnv2/dcnv2_tpu.py`，DCNv2_TPU）

复用窗口机制 + DCNv2 语义（3×3 offset conv、sigmoid mask、pos clamp 到 [0,Hp-1]、采样网格忽略 dilation、height-outer 点序、K 保留）。三档 window_radius 同 DCNv4_TPU。数值：exact 档与 flash 位级一致；窗口档窗口内 ~1.5e-6；梯度全手写且精确（CPU 对拍 ~1e-15）。

成绩（vs dcnv2_flash，128²/256² C=256）：

| config | flash fwd / fwd+bwd | dcnv2_tpu R=2 | dcnv2_tpu R=1 |
|---|---|---|---|
| 128² C=256 | 3.52 / 8.09 | 6.5 / 10.4 (0.5x/0.97x) | 4.1 / 8.4 (0.86x/1.28x) |
| 256² C=256 | 18.00 / 40.30 | 27.6 / 52.9 (0.65x/0.81x) | 18.3 / 33.5 (1.02x/1.28x) |

**结论**：dcnv2_tpu 全面超过原始 dcnv2，与 flash 持平（fwd）或略胜（fwd+bwd R=1 1.28x）；因 K-保留聚合的结构性 TPU 敌意，无法做到 DCNv4 级别的"快得多"。要在 TPU 快，DCNv2 需改走 DCNv4 结构（K 坍缩+pointwise 投影——但那就不再是 DCNv2）。

## 死路清单（TPU，全部实测）

- 客户端 jit_compile：搞崩 runtime
- tf.image.extract_patches：+59ms @128²
- 批量小 dot einsum（百万级 batch）：63ms @128²（dcnv4）/ 7213ms（dcnv2 K-保留）
- one-hot depthwise conv（multiplier=T²）：75ms fwd @128²
- K-保留聚合的所有形式（einsum/mulreduce/kloop/变体）：per-position 标量循环
- 循环累加聚合（64 次乘加替代 stack）：不融合，更差
- 分块 add_n / NCHW 布局：各 ~4-5%，不值得
- dp 物化 / while_loop adjoint：更差/不可行（pad 偏移需 Python 常量）
- separable 2-pass：per-position 权重挂在输出位置，1D 分解数学上不成立
- conv2d_backprop_input 做 col2im：one-hot filter 有 C× 冗余 → FLOP 爆炸

## 进展日志

- 2026-10-19 上午~下午：DCNv4-TPU 主线完成并合入 master（见上）。
- 2026-10-19 晚：拉取 master 合并（统一 dcnv4 路由 + dcnv2 文件夹）到 dcnv4-tpu 分支；测 dcnv2/flash TPU 基线；开发 dcnv2_tpu（数值正确，性能受 K-保留结构限制，见上）。
