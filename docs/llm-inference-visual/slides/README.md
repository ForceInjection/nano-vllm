# nano-vllm 课件（交互式网页版）

基于 [Slidev](https://sli.dev/) 的交互式课件，将 8 课教材转换为可翻页的演示文稿。

## 快速开始

```bash
cd docs/llm-inference-visual/slides

# 安装依赖（首次）
npm install

# 构建并以本地 HTTP 服务打开（推荐查看方式）
bash manage.sh serve        # → http://localhost:8080

# 或启动开发服务器（源码热更新，适合编辑课件时使用）
bash manage.sh dev           # → http://localhost:3030
bash manage.sh dev 3         # 仅预览第 3 课
```

> ⚠️ 构建产物是 SPA（单页应用），使用客户端路由如 `/1`、`/overview`。
> **不能**用 `python3 -m http.server` 等普通静态文件服务直接打开——这些服务对 `/1` 返回 404 导致空白页。
> 必须使用 `bash manage.sh serve`（内置 SPA 路由回退），或 `bash manage.sh dev`（Slidev 自带开发服务器）。

## manage.sh 命令

| 命令                         | 说明                                                  |
| ---------------------------- | ----------------------------------------------------- |
| `bash manage.sh dev`         | 启动开发服务器（全部 8 课，热更新）                   |
| `bash manage.sh dev <N>`     | 仅预览第 N 课                                         |
| `bash manage.sh build`       | 构建静态 SPA 到 `dist/`（相对路径，可部署到任意目录） |
| `bash manage.sh serve`       | 构建并用本地 HTTP 服务打开 `dist/`                    |
| `bash manage.sh clean`       | 删除 `dist/`                                          |
| `bash manage.sh clean --all` | 删除 `dist/` + `node_modules/`                        |
| `bash manage.sh lint`        | 检查标签平衡、SelfTest ID 冲突、源码引用              |
| `bash manage.sh count`       | 统计每课页数                                          |
| `bash manage.sh check-refs`  | 校验所有 `<SourceCode>` 的文件与行号                  |
| `bash manage.sh info`        | 项目概览                                              |

## 课件结构

```text
slides/
├── slides.md              # 主入口（全局配置 + 引用各课）
├── slides.01.md           # 第 1 课：LLM.generate → step 循环
├── slides.02.md           # 第 2 课：Sequence 数据结构
├── slides.03.md           # 第 3 课：Scheduler 队列与抢占
├── slides.04.md           # 第 4 课：BlockManager 与 Prefix Caching
├── slides.05.md           # 第 5 课：Prefill 批构建与 Context
├── slides.06.md           # 第 6 课：Decode 与 Block Tables
├── slides.07.md           # 第 7 课：Attention 与 KV Cache 写入
├── slides.08.md           # 第 8 课：优化全景图
├── components/            # 自定义 Vue 组件
│   ├── SelfTest.vue       #   自测题（文本展开 / 单选多选）
│   ├── SourceCode.vue     #   源码引用（带 GitHub 跳转）
│   └── ProgressWidget.vue #   学习进度追踪
├── setup/main.ts          # 全局组件注册
├── style.css              # 中文字体 + 自定义样式
├── public/                # 静态资源（背景图等）
├── manage.sh              # 管理脚本
└── dist/                  # 构建输出（gitignore）
```

## 查看构建产物

> ⚠️ **不能直接双击 `dist/index.html` 打开**。构建产物使用 ES module（`<script type="module">`），浏览器安全策略禁止 `file://` 协议加载 ES module。

正确方式：

```bash
# 方式 1：一键构建 + 启动本地 HTTP 服务（推荐）
bash manage.sh serve

# 方式 2：手动用 Python 启动 HTTP 服务
cd dist && python3 -m http.server 8080
# 浏览器打开 http://localhost:8080

# 方式 3：开发模式（源码热更新，适合编辑课件时预览）
bash manage.sh dev
```

## 每课统一结构

每课约 30+ 页，遵循统一结构：

1. **封面** — 课次 + 标题
2. **课程位置** — Mermaid 路线图，标注当前课
3. **课时安排** — 时间分配表
4. **学习目标** — 3-4 个核心问题
5. **§2 原理铺垫** — 概念说明 + 图表
6. **§3 代码走读** — 源码片段（`<SourceCode>` 标注文件行号）+ Mermaid 流程图 + 概念拆分页
7. **§4 脚本演示** — 对应 `L0x_*.py` 的 section 逐一展示
8. **课堂练习** — 可手动验证的代码
9. **课后自测题** — 3-5 道交互式问答
10. **结束页** — 知识点回顾 + 下一课预告

## 排错指南

### 构建失败

```bash
# 1. 先跑 lint 检查常见问题
bash manage.sh lint

# 2. 校验源码引用
bash manage.sh check-refs

# 3. 常见的 Vue 模板解析错误：
#    - "Attribute name cannot contain U+0022" → SelfTest answer 中混入了 ASCII 双引号
#    - "Invalid end tag" → 多余的 </div> 或标签不平衡
#    - 检查具体错误信息中的文件名，定位到对应 .md 的幻灯片

# 4. 分课排查：只构建某一课定位问题
bash manage.sh dev <N>      # 启动单课开发服务器看终端报错
```

### 开发服务器无法启动

```bash
# 重装依赖
bash manage.sh clean --all
npm install
bash manage.sh dev
```

### 新增一课

```bash
# 1. 创建 slides.09.md，参考 slides.01.md 的结构
# 2. 在 slides.md 末尾添加：
#    ---
#    src: ./slides.09.md
#    ---
# 3. 验证
bash manage.sh lint
bash manage.sh build
```

### SelfTest 组件注意事项

- `id` 格式必须为 `l{课号}-q{题号}`（如 `id="l03-q2"`），跨所有课件必须唯一
- `answer` 属性中的 HTML（如 `<strong>`、`<code>`、`<br>`）必须正确闭合
- `answer` 中**不能**使用 `<pre>` 标签（会导致 Vue 模板解析错误），用 `<code>` 替代
- `answer` 中避免 ASCII 双引号 `"`，中文引号 `「」` 可以正常使用
