---
title: "Session 2.1 Memory Abstraction & Hierarchy"
date: 2026-09-16T00:00:00+08:00
description: "GPU 内存抽象、层次结构、访存合并、共享内存与数据复用。"
categories:
  - "AI Infra"
tags:
  - "AI Infra Seminars"
  - "GPU"
hiddenFromHomePage: true
hiddenFromSearch: false
draft: false
comment: false
toc:
  enable: true
---

<div class="lecture-attribution">
  <strong>转载说明：</strong>本页内容转载自 <a href="https://infra.seminars.lcpu.dev/wiki/I4MTwyBzoi23o4kgqMLckdUdngg" target="_blank" rel="noreferrer">Infra Seminars 原始讲义</a>，作者及主讲信息见正文。原内容由北京大学学生 Linux 俱乐部与北京大学未名超算队发布，采用 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="license noreferrer">CC BY-NC-SA 4.0</a> 许可。为适配本站，仅调整了内部链接、图片路径和页面排版，正文未改写。
</div>

<div class="lecture-source">

<div class="header session-banner wiki-page-banner">

[AI Infra Wiki](/posts/infra-seminars-topic-1/)<span aria-hidden="true">/</span>[Topic 1 - Kernel and ML Compilers](/posts/infra-seminars-topic-1/)

<div class="session-banner-meta wiki-page-banner-meta">

<span class="session-banner-meta-item is-replay">**回放：**<a href="https://www.bilibili.com/video/BV1gqGA6ZEfg" target="_blank" rel="noreferrer">2.1-Memory Abstraction &amp; Hierarchy</a></span><span class="session-banner-meta-item is-presenter">**主讲：**林若瑜</span>

</div>

</div>

## GPU 的性能限制

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/c2b629ab592b8191496122bb.png" loading="lazy" decoding="async" width="2120" height="950" alt="GPU 的结构" />
</div>
<figcaption>GPU 的结构</figcaption>
</figure>

在一个 kernel 进行运算的过程中，我们需要把数据从 HBM 中移到 SM 中，然后在 SM 的运算单元中进行计算，最后在把运算结果从 SM 移回 HBM 中。这样实际上 kernel 的性能受限于计算单元的计算速度和内存带宽。

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/2b4691bb10faccd0f26c57c5.png" loading="lazy" decoding="async" width="1280" height="558" alt="性能限制 roofline model" />
</div>
<figcaption>性能限制 roofline model</figcaption>
</figure>

在 GPU 的 flops 达到一定程度以后，内存带宽满载，这时继续提升 flops 就不会带来实际性能提升。上图的拐点 I 即计算性能 / 内存带宽。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/f87b75a4d451138f2cc96f63.png" loading="lazy" decoding="async" width="1263" height="480" alt="一些典型的 GPU 的性能数据" />
</div>
<figcaption>一些典型的 GPU 的性能数据</figcaption>
</figure>

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/e6409c6733cf9862f6aff725.png" loading="lazy" decoding="async" width="1534" height="280" alt="常见 kernel 的算存比" />
</div>
<figcaption>常见 kernel 的算存比</figcaption>
</figure>

可以看出，以上 kernel 中，GEMM 是一个 compute bound 的 kernel，其余均为 memory bound。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.446195;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/8f2adf22e2fc98ad028e88de.png" loading="lazy" decoding="async" width="1178" height="484" alt="GEMM" />
</div>
<figcaption>GEMM</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.553805;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/b57b97f54dce663a8f1b00db.png" loading="lazy" decoding="async" width="1206" height="398" />
</div>
</figure>

</div>

</div>

## 用满内存带宽

对于 roofline model给出的上限性能，在实际情况中也是不容易达到的。下面将介绍如何打满内存的带宽，从而优化 kernel 性能。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
template <typename scalar_t>
__global__ void axpy_naive(scalar_t a, const scalar_t * x, scalar_t * y, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        y[tid] = a * x[tid] + y[tid];
    }
}
```

</div>

上图 kernel 有完美的 coalesced、没有divergence，没有同步、100% occupancy 而且 embarrassingly parallel，没有数据依赖。但在实际情况下，这样的 kernel 也无法跑满内存带宽。下图显示，使用不同的数据类型，这个 kernel 的内存带宽利用率有明显区别，而且和显卡的最大内存带宽（stream）也有明显差距。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/a140e7c4b034a7eb7927ef8c.png" loading="lazy" decoding="async" width="1688" height="990" alt="kernel 在不同显卡、不同数据类型时的内存带宽利用率" />
</div>
<figcaption>kernel 在不同显卡、不同数据类型时的内存带宽利用率</figcaption>
</figure>

我们这样来分析上述的问题：想象有一个20级台阶的扶梯，扶梯的每个台阶站1个人，扶梯运行的速度是2s一个台阶，那么我们可以算出这个扶梯的带宽是0.5人/s，延迟是40s。但是如果整个扶梯上只有一个人，那么相当于在40s的延迟里，扶梯一共才运输了一个人，这时的实际带宽只有1/40 人/s，而且扶梯上的位置大部分也是空的。我们需要带宽\*延迟，也就是20个人站在扶梯上，才能跑满这个扶梯的理论带宽。（**Little's law** in Queueing Theory）

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.479345;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/843d340efc5e0812a0e4db8d.png" loading="lazy" decoding="async" width="1105" height="945" alt="只有一个人，带宽没有跑满" />
</div>
<figcaption>只有一个人，带宽没有跑满</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.520655;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/7e72e0e60d3f12bef7441bb1.png" loading="lazy" decoding="async" width="796" height="626" />
</div>
</figure>

</div>

</div>

所以 GPU 中，我们需要 带宽\*延迟 这个数量的 infly 字节（即在请求队列里），下表给出了常见 GPU 的带宽和延迟数（估计），从而可以知道我们需要的 infly 字节数。

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/3a8aeab8d6c74272accc4b3d.png" loading="lazy" decoding="async" width="1491" height="736" />
</div>
</figure>

如果想提高在飞字节数，一般有两种方法：instruction level parallelism 和 thread level parallelism。

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/04b6dca0bfb5ccbaffaf5762.png" loading="lazy" decoding="async" width="2253" height="850" alt="提高在飞字节数的方法" />
</div>
<figcaption>提高在飞字节数的方法</figcaption>
</figure>

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/e12449004519e9b3cb3aee9a.png" loading="lazy" decoding="async" width="2445" height="304" alt="提升上述三个指标都可以提升在飞字节数" />
</div>
<figcaption>提升上述三个指标都可以提升在飞字节数</figcaption>
</figure>

### 提高在飞字节数——展开

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__
void kernel(const float * __restrict__ a,
            const float * __restrict__ b,
            float * __restrict__ c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    c[idx] += a[idx] * b[idx];
}
```

</div>

我们假设 load 指令的延迟是1000个 cycle，每个 cycle 可以发出1条指令，那么在执行完前2个 cycle 时（三个 load 指令），之后我们需要等待1000个 cycle的数据才能等到计算开始！这样一次循环需要1006个 cycle。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.416415;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/b1830367f867129b265a230f.png" loading="lazy" decoding="async" width="1475" height="675" alt="指令化的 kernel" />
</div>
<figcaption>指令化的 kernel</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.583585;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/b1830367f867129b265a230f.png" loading="lazy" decoding="async" width="1475" height="675" alt="我们需要等待1000个 cycle！" />
</div>
<figcaption>我们需要等待1000个 cycle！</figcaption>
</figure>

</div>

</div>

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__
void kernel(const float * __restrict__ a,
            const float * __restrict__ b,
            float * __restrict__ c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll 2
    for (int i = 0; i < 2; i++) {
        const int idx = tid + i * stride;
        c[idx] += a[idx] * b[idx];
    }
}
```

</div>

对于这个修改后的 kernel，编译器虽然可以直接判断 a 和 b 的项没有修改，可以直接在 `load c[i1]` 之后执行 `load a[i2]` 和 `load b[i2]`。但是因为对 c 做了修改，而且编译器不能确定 `stride` 是一个编译期常量，如果 `stride = 0`，那么对 c 的修改只能顺序执行。所以编译器会卡在 `load c[i2]` 之前。这样两次循环总共需要2011个 cycle，一个循环比上面节省了0.5个cycle。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.40164;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/15ab470edbdddb89541ee2df.png" loading="lazy" decoding="async" width="1242" height="736" alt="指令化的kernel" />
</div>
<figcaption>指令化的kernel</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.59836;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/80d8f7c31d803793f188236a.png" loading="lazy" decoding="async" width="1688" height="668" />
</div>
</figure>

</div>

</div>

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
#define THREAD_BLOCK_DIM 128

__global__
void kernel(const float * __restrict__ a,
            const float * __restrict__ b,
            float * __restrict__ c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int off = 2 * THREAD_BLOCK_DIM * blockIdx.x + threadIdx.x;

    #pragma unroll 2
    for (int i = 0; i < 2; i++) {
        const int idx = off + i * THREAD_BLOCK_DIM;
        c[idx] += a[idx] * b[idx];
    }
}
```

</div>

在这次修改中，我们使用了编译期常量，才将问题得到解决。这次，我们在1008个 cycle 中执行了2个循环，相比之下效率提升了2倍。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.438473;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/64f68ba54fcf145d280b364c.png" loading="lazy" decoding="async" width="1218" height="682" alt="可以提前执行所有 load" />
</div>
<figcaption>可以提前执行所有 load</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.561527;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/5834e762ce2d959e20f913d3.png" loading="lazy" decoding="async" width="1528" height="666" />
</div>
</figure>

</div>

</div>

### 提高内存带宽——Occupancy

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/3b4c0bf1b82a76f968b65da9.png" loading="lazy" decoding="async" width="799" height="1118" alt="SM结构示意图" />
</div>
<figcaption>SM结构示意图</figcaption>
</figure>

一个 SM 内部有4个 SMSP， 每个 SMSP 都有自己独立的算术单元、寄存器和一个 warp scheduler。一个 SMSP 上最多驻留16个 warp，每个 cycle 都可以选择一个 warp 从中发射。因为每个 SMSP 有自己的寄存器，所以不需要类似 CPU 的上下文切换。所以理论上发射的 warp 越多，occupancy 越高，理论上内存带宽利用率就越高。

Shared memory 的大小和寄存器数量会限制 occupancy。shared memory 的典型大小是每个 SM 164KB（A100）或228KB（H100、B200）；每个 SM 有65536个32 bit 寄存器。

下图呈现了 A100 上实测的 occupancy 和 MLP 对于内存带宽的影响。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/a60f1388026809b6cd5f3449.png" loading="lazy" decoding="async" width="962" height="572" alt="A100 的实测内存带宽" />
</div>
<figcaption>A100 的实测内存带宽</figcaption>
</figure>

### 提高内存带宽——向量化访存

我们可以使用 `float4` 类型，使得一次 load 可以加载4个 float（128 bytes）。相当于每一个 load 指令的效率都提高了4倍。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
// x and y must be 16-byte aligned (cudaMalloc allocations satisfy this).
__global__ void saxpy_vectorized(const float4* __restrict__ x,
                                 float4* __restrict__ y, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = n / 4;

    if (i < n4) {
        const float4 xv = x[i];
        const float4 yv = y[i];

        y[i] = make_float4(a * xv.x + yv.x,
                           a * xv.y + yv.y,
                           a * xv.z + yv.z,
                           a * xv.w + yv.w);
    }
}
```

</div>

### 提高内存带宽——coalescing

多个线程的内存访问尽可能合并（Coalescing），即让一个 warp 中多个线程访问连续地址的数据，可以减少实际的内存事务数量。

GPU 的内存访问具有固定的粒度。以 NVIDIA GPU 为例，最小的内存访问单位通常是 32 Bytes 的 sector。从 L1 Cache 到 L2 Cache 时，一个 sector 为 32 Bytes；而从 L2 Cache 到 Global Memory 时，通常以多个 sector 组成更大的访问粒度。Cache 管理的基本单位则是 128 Bytes 的 Cache Line，一个 Cache Line 由 4 个连续的 32 Bytes sector 组成。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/442f35cfa5361aa3b4ec2173.png" loading="lazy" decoding="async" width="1170" height="296" alt="cache line" />
</div>
<figcaption>cache line</figcaption>
</figure>

我们要尽量让一个 warp 读取的数据位于连续4个 sector 的内存上：

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.50007;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/5b47f2912c9482c1f02d62ba.png" loading="lazy" decoding="async" width="1171" height="555" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.49993;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/8db2b327f0df802f34433300.png" loading="lazy" decoding="async" width="1042" height="494" />
</div>
</figure>

</div>

</div>

如果没有以32个 byte 对齐、读取集中在一个 sector 或数据离得很远，都会影响内存效率。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.346019;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/8d5e23b7c05132857ba831be.png" loading="lazy" decoding="async" width="944" height="402" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.314876;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/d1df0918792f5a767debd976.png" loading="lazy" decoding="async" width="1026" height="481" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.339105;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/6d2689a80e7ec4509a45aa30.png" loading="lazy" decoding="async" width="957" height="416" />
</div>
</figure>

</div>

</div>

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
struct Coefficients
{
    float u, v, w;
    float x[8], y[8], z;
};

__global__ void kernel(Coefficients *data)
{
    int i = cg::this_grid().thread_rank();

    data[i].u = data[i].u + 10.f;
    data[i].y[0] = data[i].y[0] + 10.f;
}
```

</div>

例如把数据放到一个结构体里，每一个结构体里的 u 会离得很远。第11行对 u 的访问操作就会跨不同的 sector，效率低下。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/93b3b599a734b921be1a1f24.png" loading="lazy" decoding="async" width="1650" height="502" alt="对 u 的访问跨越了不同 sector" />
</div>
<figcaption>对 u 的访问跨越了不同 sector</figcaption>
</figure>

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
struct Coefficients
{
    float *u, *v, *w;
    float *x0, ..., *x7, *y0, ..., *y7, *z;
};

__global__ void kernel(Coefficients data)
{
    int i = cg::this_grid().thread_rank();

    data.u[i] = data.u[i] + 10.f;
    data.y0[i] = data.y0[i] + 10.f;
}
```

</div>

但是如果稍作修改，把原先的 struct 改成 struct of arrays，那么可以看到访问的内存就便连续了，实际测试中可以提升大约8倍的性能。

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/ca872f6f5e5ad97706db4bd3.png" loading="lazy" decoding="async" width="1530" height="584" alt="u 位于连续的内存上" />
</div>
<figcaption>u 位于连续的内存上</figcaption>
</figure>

### 例子：优化 reduce

#### 朴素的串行相加

对于给定长度 N 的数组，对其中每项进行求和：$`out = \sum\limits_{i = 0}^{N - 1}x_{i}`$。我们将以这个算法为例，给大家展示如何进行优化。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v1(
    const float* x, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        atomicAdd(out, x[i]);
}
```

</div>

第一个版本的 reduce 中，所有变量朴素的通过一个循环相加。这样所有的请求都会串行的发送给 L2 slice。实测的执行时间是 197ms，内存带宽是1.4GB/s。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.558921;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/99d464c7226be4370c4f28d4.png" loading="lazy" decoding="async" width="961" height="541" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.441079;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/0bffef01d0e26f652e4d77ee.png" loading="lazy" decoding="async" width="1181" height="845" />
</div>
</figure>

</div>

</div>

#### 使用 shared memory 和树形优化

我们可以把输入数组放到 shared memory 中，防止重复访问 global memory；每个 block 可以在内部进行一个树形的 reduce 操作，优化后的reduce：

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v2(
    const float* x, float* out, int n)
{
    __shared__ float s[BLOCK];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (tid % (2*stride) == 0)
            s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, s[0]);
}
```

</div>

这样经过实际测试，一次 reduce 耗时1.67ms，内存带宽 160GB/s，执行效率提高了118倍。但是可以看的，并不是 warp 里连续的线程在执行操作，也就是 warp 中一半的线程被浪费了。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.36236;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/4bfa3fe802db51a02b870cef.png" loading="lazy" decoding="async" width="1953" height="1778" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.63764;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/aee529f045cf25e7d7bdfdc9.png" loading="lazy" decoding="async" width="1406" height="722" />
</div>
</figure>

</div>

</div>

#### 使得 warp 中的线程连续

再次优化后，我们让连续的线程负责相邻的相加。这样可以再次得到2倍的加速。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v3(
    const float* x, float* out, int n)
{
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    s[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int idx = 2 * stride * tid;
        if (idx < blockDim.x)
            s[idx] += s[idx + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, s[0]);
}
```

</div>

### 提高内存带宽——bank conflict

Bank 是 shared memory 中可以独立进行一次访问的最小存储分区。GPU 将 shared memory 划分为多个 bank，每个 bank 具有独立的数据通路；当一个 warp 中不同线程访问不同 bank 时，可以并行完成访问，而多个线程访问同一个 bank 时会产生 bank conflict。

在 NVIDIA GPU 中，一个 SM 的 shared memory 通常划分为 32 个 bank，把 byte 地址按照4字节分块，连续的4个字节属于一个 bank，而且128个字节后的四个字节也属于这个 bank。例如下图中，地址0-3、128-121、256-259的字节属于 bank0，地址4-7、132-135、260-263属于 bank1……

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.486094;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/c73c75391df16e194439360d.png" loading="lazy" decoding="async" width="1584" height="704" alt="bank 的划分" />
</div>
<figcaption>bank 的划分</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.513906;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/7781e78c67bdb09ebdfc1c37.png" loading="lazy" decoding="async" width="1821" height="765" alt="bank conflict" />
</div>
<figcaption>bank conflict</figcaption>
</figure>

</div>

</div>

### 例子：优化 reduce

#### 解决 bank conflict

在刚刚的 reduce kernel 中，前面的迭代轮次里读取的字节都是连续的，不存在 bank conflict。但是靠后的轮次中，需要间隔好几个 byte 才能读取数据，造成 bank conflict，导致线程串行化。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v4(const float* __restrict__ x, float* out, int n) {
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? x[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, s[0]);
}
```

</div>

修改后的代码可以使得每个 warp 访问的数据都是连续的。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.207372;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/45d2240ad20ef79a23e3ff64.png" loading="lazy" decoding="async" width="1952" height="1773" alt="内存访问都是连续的" />
</div>
<figcaption>内存访问都是连续的</figcaption>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.792628;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/851da024ad9d3fd2ebd61154.png" loading="lazy" decoding="async" width="574" height="133" />
</div>
</figure>

</div>

</div>

这样可以把执行效率提升到 0.78ms，344 GB/s，相对之前又加快了1.9倍。但是分析仍然表明，这样的kernel DRAM 利用率偏低，说明从之前的 memory bound 变成了 compute bound。

#### 控制 block 数量

分析代码可以知道，现在的 block 数量是和数据长度有关的。在执行过程中，我们因为 block 过多，执行了很多不必要的 reduce 操作，因此降低了执行效率。我们把一个 thread 执行多个操作，延迟 thread 的生命周期，并减少 block 数量，从而又进行了一次优化：

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v5(const float* __restrict__ x, float* out, int n)
{
    __shared__ float s[BLOCK];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        sum += x[i];

    s[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, s[0]);
}
```

</div>

这让执行时间缩短到了 0.25ms，带宽提升到1074GB/s。

#### 在任务规模足够小时使用寄存器

如果观察到当任务折半到 stride ≤ 16时，可以发现所有 thread 都位于一个 thread，这时候实际上是不需要动用 shared memory，因为 warp 中的32个 lane 可以直接使用寄存器。例如：

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}
```

</div>

| 指令 | lane i 接收 | 用途 |
|----|----|----|
| `__shfl_sync` | lane: `srcLane`（可指定） | 广播 / 置换 |
| `__shfl_up_sync` | lane: `i - delta` | Prefix sum, stencil |
| `__shfl_down_sync` | lane: `i + delta` | Reduce（归约到 lane 0） |
| `__shfl_xor_sync` | lane: `i xor laneMask` | Butterfly Reduce（交换到所有 lane） |

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/c59a00e3e6b1813d806fe0d5.png" loading="lazy" decoding="async" width="624" height="183" alt="shuffle 指令" />
</div>
<figcaption>shuffle 指令</figcaption>
</figure>

上面的 reduce 操作也可以使用 `__shfl_down_sync` 实现，二者效率类似。

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v6(const float* __restrict__ x, float* out, int n)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        sum += x[i];

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[32];

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sum[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (lane < blockDim.x / 32) ? warp_sum[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}
```

</div>

通过这样的优化，可以把执行时间为0.247ms，带宽1087GB/s。但是可以发现，这样的优化并没有带来显著的性能提升。这是因为这时 kernel 达到了 memory bound，需要进一步优化。

#### 向量化访存

我们通过使用 `float4` 进一步提升内存带宽：

<div class="language-C++ vp-adaptive-theme">

<span class="lang">C++</span>

```text
__global__ void reduce_v7(const float* __restrict__ x4, float* out, int n4)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
    {
        float4 v = x4[i];
        sum += v.x + v.y + v.z + v.w;
    }

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sum[32];

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_sum[wid] = sum;
    __syncthreads();

    if (wid == 0) {
        sum = (lane < blockDim.x / 32) ? warp_sum[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) atomicAdd(out, sum);
    }
}
```

</div>

最终把执行时间压缩到0.17ms，内存带宽可达1572GB/s。

#### 小结

| 版本             | 时间        | 带宽          | vs V1     | warp 指令  |
|------------------|-------------|---------------|-----------|------------|
| V1 global atomic | 197.4 ms    | 1.4 GB/s      | 1×        | —          |
| V2 smem，散开    | 1.67 ms     | 161 GB/s      | 118×      | 1.55 亿    |
| V3 连续线程      | 0.87 ms     | 308 GB/s      | 227×      | 6167 万    |
| V4 sequential    | 0.78 ms     | 344 GB/s      | 253×      | 5276 万    |
| V5 grid-stride   | 0.25 ms     | 1074 GB/s     | 790×      | 560 万     |
| V6 warp shuffle  | 0.247 ms    | 1087 GB/s     | 799×      | 460 万     |
| **V7 + float4**  | **0.17 ms** | **1572 GB/s** | **1156×** | **​185 万** |

<figure class="feishu-figure">
<div class="is-transparent feishu-image-frame">
<img src="/lectures/topic-1/media/I4MTwyBzoi23o4kgqMLckdUdngg/db13e3c60163e868c2b1e8a3.png" loading="lazy" decoding="async" width="1493" height="509" />
</div>
</figure>

可以看到优化效果明显。不过值得注意的是，如果对v5使用 `float4` 优化，其最终性能和 v7 相仿，说明使用 warp shuffle 优化效果目前不明显。

</div>
