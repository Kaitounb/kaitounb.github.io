---
title: "Session 3 Tensor Core"
date: 2026-09-16T00:00:00+08:00
description: "Tensor Core、MMA tile shape 与布局相关的课前资料。"
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
  <strong>转载说明：</strong>本页内容转载自 <a href="https://infra.seminars.lcpu.dev/wiki/QsOcw3uHpifm10k5ouCcK6Uonee" target="_blank" rel="noreferrer">Infra Seminars 原始讲义</a>，作者及主讲信息见正文。原内容由北京大学学生 Linux 俱乐部与北京大学未名超算队发布，采用 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank" rel="license noreferrer">CC BY-NC-SA 4.0</a> 许可。为适配本站，仅调整了内部链接、图片路径和页面排版，正文未改写。
</div>

<div class="lecture-source">

<div class="header session-banner wiki-page-banner">

[AI Infra Wiki](/posts/infra-seminars-topic-1/)<span aria-hidden="true">/</span>[Topic 1 - Kernel and ML Compilers](/posts/infra-seminars-topic-1/)

<div class="session-banner-meta wiki-page-banner-meta">

<span class="session-banner-meta-item is-presenter">**主讲：**孙远航</span>

</div>

</div>

## 课前阅读

- <a href="https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma" target="_blank" rel="noreferrer">https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma</a> 只需要看m16n8k16 那一节，重点是 A 和 B 的 fragment 布局
- <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory" target="_blank" rel="noreferrer">https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory</a> 复习 bank 的划分和 conflict 的成因
- DeepSeek-V3 Technical Report，只读 FP8 训练小节

</div>
