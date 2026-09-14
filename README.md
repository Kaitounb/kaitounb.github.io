# Kaitou 的博客

一个使用 [Hugo](https://gohugo.io/) 与 [FixIt](https://github.com/hugo-fixit/FixIt) 构建的个人博客。源码由 GitHub 管理，GitHub Actions 自动构建，GitHub Pages 托管静态文件，不需要服务器。

## 本地预览

```bash
hugo server --environment production --disableFastRender
```

打开 `http://localhost:1313/`。

## 写一篇新文章

```bash
hugo new content posts/my-new-post.md
```

文章在 `content/posts/`，使用 Markdown 编写。确认 front matter 里的 `date`、`categories` 和 `tags` 后即可发布。

## GitHub Pages 部署

仓库建议命名为 `kaitounb.github.io`。推送到 `main` 后，`.github/workflows/hugo.yaml` 会自动构建并发布。

首次发布需要在 GitHub 仓库中进入 **Settings → Pages**，将 **Source** 设为 **GitHub Actions**。

## 绑定 Cloudflare 域名

以 `blog.example.com` 为例：

1. 先在 GitHub 的 **Settings → Pages → Custom domain** 填入 `blog.example.com`。
2. 在 Cloudflare DNS 添加 `CNAME`：名称 `blog`，目标 `kaitounb.github.io`。
3. 域名校验和 GitHub HTTPS 证书签发完成前，先使用 **DNS only**（灰云）。
4. GitHub Pages 显示 HTTPS 可用后，再切换为 **Proxied**（橙云）。
5. Cloudflare 的 SSL/TLS 模式使用 **Full (strict)**，并开启 **Always Use HTTPS**。

> 阿里云只负责域名注册。若把 DNS 托管交给 Cloudflare，需要在阿里云域名控制台把 nameserver 改成 Cloudflare 分配的两个地址。

## 主要目录

```text
content/                 Markdown 文章与页面
layouts/home.html        自定义首页
assets/css/              视觉样式
config/                  Hugo 与 FixIt 配置
.github/workflows/       GitHub Pages 自动部署
```
