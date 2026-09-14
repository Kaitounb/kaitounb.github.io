---
title: "Stage 0：在测性能之前，先把数字测对"
date: 2026-09-14T17:00:00+08:00
description: "用 RTX 3070 做一次最小 torch.matmul 实验，弄清 warm-up、CUDA 异步计时、p50/p95，以及 allocated、reserved 和 nvidia-smi 的区别。"
categories:
  - "AI Infra"
tags:
  - "GPU"
  - "CUDA"
  - "PyTorch"
  - "Benchmark"
draft: false
comment: false
---

Stage 0 只回答一个问题：**我测到的数字，到底是什么？**

它可能是 GPU 真正计算的时间，也可能只是 CPU 把任务交出去的时间；可能混进了第一次运行的初始化，也可能只是 PyTorch 留着没还给驱动的显存。数字本身不会告诉我这些，实验边界才会。

## 开始之前：先补一点底层直觉

我最开始想直接理解 CPU 和 GPU 的分工，但很快发现，中间还缺一段“晶体管为什么能执行指令”的桥。

我在 NandGame 里用两个 relay 搭出了一个 NAND：先用一个开关得到 `a AND b`，再用另一个开关把结果取反，四组输入都通过。真正卡住我的不是 NAND 公式，而是英文界面里的 relay、control、input 和各种线路符号。继续硬接线只会同时猜英文、猜符号、学电路，所以我先暂停了这条支线。

Stage 0 暂时采用一个够用的模型：CPU 把工作提交给 GPU 后可以继续往下走；`torch.cuda.synchronize()` 会让 CPU 等到 GPU 做完。更底层的 CPU 指令执行、操作系统调度和 CUDA 实现，后面遇到问题时再补。

## 实验环境

- Windows 10
- NVIDIA RTX 3070 8GB
- NVIDIA Driver 591.74
- Python 3.13.5
- PyTorch 2.10.0+cu128
- PyTorch CUDA Runtime 12.8
- `4096 × 4096` FP32 矩阵乘法
- 10 次 warm-up，之后正式测量 30 次
- 相同配置启动三个独立进程重复实验

Windows 机器不能稳定访问海外下载源，最后用阿里云镜像装好了 CUDA 版 PyTorch。这里没有安装 `nvcc`，因为只是运行 PyTorch 已经编译好的 CUDA wheel，还没有写自己的 CUDA 源码。

## 第一次为什么慢

三个独立进程里，第一个 `matmul` 分别用了 57.272、54.915 和 58.957 ms；warm-up 后的稳态 p50 只有约 12.33 ms。第一次是稳态的 4.45～4.78 倍。

| Run | 第一个 matmul | 稳态 p50 | 稳态 p95 |
|---|---:|---:|---:|
| 1 | 57.272 ms | 12.329 ms | 12.348 ms |
| 2 | 54.915 ms | 12.334 ms | 12.342 ms |
| 3 | 58.957 ms | 12.328 ms | 12.345 ms |

这次输入矩阵和输出矩阵在计时前已经直接创建在 GPU 上，并且做过同步，所以这 57 ms 不包含 CPU 到 GPU 的数据搬运。它主要说明第一次执行仍然带着一次性的初始化和准备成本。

我之前容易把 warm-up 理解成“程序第一次必须完成的一项工作”。更准确的说法是：**warm-up 是我主动先跑几次，并把这些样本丢掉，避免一次性成本污染稳态结果。**

三次稳态 p50 的极差只有 0.0066 ms，约为中位值的 0.054%。至少在当前环境和参数下，这个 baseline 是稳定的。

## 0.035 ms 的假象

下面两种写法只差一次同步：

```python
# 错误：GPU 可能还没算完，CPU 秒表已经停了
start = time.perf_counter()
torch.matmul(a, b, out=out)
wrong_ms = (time.perf_counter() - start) * 1000

# 正确：停止计时前等待 GPU 完成
start = time.perf_counter()
torch.matmul(a, b, out=out)
torch.cuda.synchronize()
correct_ms = (time.perf_counter() - start) * 1000
```

未同步时，三次 p50 约为 0.035 ms；同步后约为 12.462 ms，差了 356～359 倍。

0.035 ms 主要测到的是 CPU 发起并提交 GPU 工作的时间。函数返回时，GPU 还在计算。只有在秒表停止前等待 GPU 完成，墙钟时间才覆盖了这次工作。

这也是我在 Stage 0 最重要的收获：**代码执行到了下一行，不等于 GPU 已经完成了上一行的计算。** PyTorch 的 [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution) 和 [`torch.cuda.synchronize`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html) 文档给出了准确边界。

## CUDA Event 和 CPU 秒表在测什么

我又用 CUDA Event 对同一次矩阵乘法做了交叉验证：

| Run | 同步 CPU 秒表 | CUDA Event |
|---|---:|---:|
| 1 | 12.810 ms | 12.251 ms |
| 2 | 12.302 ms | 12.130 ms |
| 3 | 12.431 ms | 12.250 ms |
| p50 | 12.431 ms | 12.250 ms |

我的预测是 CUDA Event 应该接近同步后的 12 ms，而不是未同步的 0.035 ms，结果确实如此。

两者不需要完全一样。CUDA Event 记录的是 GPU 工作队列中两个事件之间的时间；CPU 秒表还会覆盖 Python 调用、CUDA API 调用以及同步返回等边界开销。这次结果也不包含 GPU 到 CPU 的结果传输，因为输出一直留在 GPU 上，没有调用 `.cpu()` 或 `.to("cpu")`。

## 删除 Tensor 后，显存为什么没有立刻下降

我创建了一个 512 MiB 的 FP32 Tensor，分别观察 PyTorch 和 `nvidia-smi`：

| 阶段 | allocated | reserved | nvidia-smi 整卡 used |
|---|---:|---:|---:|
| 创建前 | 8.125 MiB | 20 MiB | 1070 MiB |
| 创建后 | 520.125 MiB | 532 MiB | 1606 MiB |
| `del` + `gc.collect()` 后 | 8.125 MiB | 532 MiB | 1606 MiB |
| `torch.cuda.empty_cache()` 后 | 8.125 MiB | 20 MiB | 1094 MiB |

现在我对这三个数的理解是：

- `allocated`：当前还被活跃 Tensor 使用的显存。
- `reserved`：PyTorch 已经从 CUDA 申请并留在缓存池里的显存，其中可以包含暂时没有 Tensor 使用的空间。
- `nvidia-smi`：从驱动和整张 GPU 的角度观察，除了 Tensor，还可能包含 CUDA context、框架和桌面显示等占用。

所以删除 Tensor 后，`allocated` 下降了 512 MiB，但 PyTorch 暂时保留这块空间，`reserved` 没有变化。调用 `empty_cache()` 后，未使用的缓存归还给驱动，`reserved` 和 `nvidia-smi` 才下降。PyTorch 的 [CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management) 也提醒，缓存分配器中未使用的显存仍可能在 `nvidia-smi` 里显示为已占用。

## p50、p95 和最大值

我以前容易把 p95 当成平均值或者“接近最大值”。现在的理解是：

- p50 是中位数，一半样本不超过它。
- p95 表示大约 95% 的样本不超过它，用来看偏慢的尾部。
- 最大值只是这批样本里最慢的一次，很容易被偶发事件影响。

这次每组只有 30 个样本，95% 对应第 28.5 个位置，不可能真的取“半个样本”，所以不同统计实现会选择邻近位置或进行插值。p95 是基于有限样本的估计，不是某个天然存在的精确点。

## Stage 0 之后，我至少不会再犯这些错

1. 不把第一次运行和稳态结果混在一起。
2. 不用未同步的 CPU 秒表冒充 GPU 计算时间。
3. 不只跑一次就下结论，要重复测量并报告 p50/p95。
4. 不把 Tensor 大小、PyTorch reserved 和 `nvidia-smi` 当成同一个口径。
5. 写清楚矩阵大小、dtype、warm-up、重复次数、同步位置和计时边界。

Stage 0 的价值不是记住 12.33 ms，而是以后看到任何性能数字时，先问一句：**它到底测了什么，又漏掉了什么？**

下一站是 Stage 1：一个 token 是怎样生成的。

## 参考

- [PyTorch CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [torch.cuda.synchronize](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)
- [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark)
