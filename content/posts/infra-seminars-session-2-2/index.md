---
title: "Session 2.2 FP32 GEMM Quick Walkthrough"
date: 2026-09-16T00:00:00+08:00
description: "FP32 GEMM 优化路径快速导览。"
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
  <strong>转载说明：</strong>本页内容转载自 <a href="https://infra.seminars.lcpu.dev/wiki/AwWTwMUDxiO5APkFt2Jc9hhknsf" target="_blank" rel="noreferrer">Infra Seminars 原始讲义</a>，作者及主讲信息见正文。原内容由北京大学学生 Linux 俱乐部与北京大学未名超算队发布，采用 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="license noreferrer">CC BY-NC-SA 4.0</a> 许可。为适配本站，仅调整了内部链接、图片路径和页面排版，正文未改写。
</div>

<div class="lecture-source">

<div class="header session-banner wiki-page-banner">

[AI Infra Wiki](/posts/infra-seminars-topic-1/)<span aria-hidden="true">/</span>[Topic 1 - Kernel and ML Compilers](/posts/infra-seminars-topic-1/)

<div class="session-banner-meta wiki-page-banner-meta">

<span class="session-banner-meta-item is-replay">**回放：**<a href="https://www.bilibili.com/video/BV1L8GA6YEAH" target="_blank" rel="noreferrer">2.2-FP32 GEMM Quick Walkthrough</a></span><span class="session-banner-meta-item is-presenter">**主讲：**周宇轩</span>

</div>

</div>

## 矩阵乘法 （GEMM）

计算$`C = A \times B,A \in \mathbb{R}^{M \times K},B \in \mathbb{R}^{K \times N},C \in \mathbb{R}^{M \times N}`$

我们如何计算这样的矩阵乘法呢？按照并行计算的思路，我们需要拆分计算和数据。我们需要使用数据并行的思路进行数据拆分，把数据拆分到 thread 尺度。

在传统的深度学习中，我们一般使用 FP16 甚至 FP8 这类低精度的运算，但是科学计算中，仍然会涉及到32位精度浮点数的矩阵乘法。这时我们没有办法使用矩阵运算的专用硬件 tensor core，需要自己设计一个并行化的 kernel。

<div class="feishu-grid">

<div class="feishu-grid-column" style="--feishu-column-width:0.469502;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/AwWTwMUDxiO5APkFt2Jc9hhknsf/4fe2fde86a589a2f311e9e51.png" loading="lazy" decoding="async" width="956" height="888" />
</div>
</figure>

</div>

<div class="feishu-grid-column" style="--feishu-column-width:0.530498;">

<figure class="feishu-figure">
<div class="feishu-image-frame">
<img src="/lectures/topic-1/media/AwWTwMUDxiO5APkFt2Jc9hhknsf/e42635b554c395f31b14f0b3.png" loading="lazy" decoding="async" width="1464" height="1202" />
</div>
</figure>

</div>

</div>

</div>
