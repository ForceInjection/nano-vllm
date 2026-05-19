#!/usr/bin/env bash
# ============================================================
# nano-vllm 课件管理脚本
# 用法: bash manage.sh <command> [args]
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

# ── 帮助 ──────────────────────────────────────────────
help() {
  cat << 'EOF'
nano-vllm 课件管理脚本

用法: bash manage.sh <command> [args]

═══════════════════════════════════════════════════════
  开发
═══════════════════════════════════════════════════════
  dev                 启动开发服务器（全部 8 课）
  dev <N>             仅预览第 N 课（如: bash manage.sh dev 3）
  build               构建全部课件到 dist/
  serve               构建并用本地 HTTP 服务打开 dist/（用于验证构建产物）
  clean               清理 dist/ 和 node_modules/

═══════════════════════════════════════════════════════
  检查与诊断
═══════════════════════════════════════════════════════
  lint                检查常见问题（标签平衡、SelfTest ID 冲突等）
  count               统计每课幻灯片页数
  check-refs          校验所有 <SourceCode> 引用的文件与行号是否存在

═══════════════════════════════════════════════════════
  其他
═══════════════════════════════════════════════════════
  help                显示本帮助
  info                显示项目概览（总页数、依赖版本等）
EOF
}

# ── 开发服务器 ────────────────────────────────────────
dev() {
  if [[ $# -gt 0 && "$1" =~ ^[0-8]$ ]]; then
    local n="$1"
    if [[ ! -f "slides.0${n}.md" ]]; then
      echo -e "${RED}错误: slides.0${n}.md 不存在${NC}"
      exit 1
    fi
    # 创建临时入口只引用单课
    cat > .tmp_slides.md << MD
---
theme: default
title: nano-vllm 第${n}课
---
---
src: ./slides.0${n}.md
---
MD
    echo -e "${CYAN}▶ 仅预览第 ${n} 课...${NC}"
    npx slidev --entry .tmp_slides.md
    rm -f .tmp_slides.md
  else
    echo -e "${CYAN}▶ 启动开发服务器（全部 8 课）...${NC}"
    npx slidev
  fi
}

# ── 构建 ──────────────────────────────────────────────
build() {
  echo -e "${CYAN}▶ 构建全部课件（相对路径模式，生成的文件可直接用浏览器打开）...${NC}"
  npx slidev build --out dist --base ./
  local size
  size=$(du -sh dist/ 2>/dev/null | cut -f1)
  echo -e "${GREEN}✓ 构建完成${NC} (输出: dist/, 大小: ${size})"
}

# ── 本地 HTTP 服务 ────────────────────────────────────
serve() {
  if [[ ! -d dist/ ]] || [[ ! -f dist/index.html ]]; then
    echo -e "${YELLOW}dist/ 不存在或未构建，先执行 build...${NC}"
    build
  fi
  local port="${1:-8080}"
  echo -e "${GREEN}▶ 启动 SPA HTTP 服务（支持客户端路由，如 /1、/overview）...${NC}"
  echo -e "${GREEN}  打开浏览器访问: ${CYAN}http://localhost:${port}${NC}"
  echo -e "${YELLOW}  按 Ctrl+C 停止${NC}"
  echo ""
  if command -v python3 &>/dev/null; then
    python3 serve_spa.py "$port" dist
  elif command -v python &>/dev/null; then
    python serve_spa.py "$port" dist
  else
    echo -e "${RED}错误: 未找到 python3，请安装 Python${NC}"
    exit 1
  fi
}

# ── 清理 ──────────────────────────────────────────────
clean() {
  echo -e "${YELLOW}▶ 清理 dist/ ...${NC}"
  rm -rf dist/
  echo -e "${GREEN}✓ dist/ 已删除${NC}"
  if [[ "${1:-}" == "--all" ]]; then
    echo -e "${YELLOW}▶ 清理 node_modules/ ...${NC}"
    rm -rf node_modules/
    echo -e "${GREEN}✓ node_modules/ 已删除${NC}"
    echo -e "${YELLOW}提示: 重新安装请运行 ${CYAN}npm install${NC}"
  fi
}

# ── 幻灯片计数 ────────────────────────────────────────
count() {
  echo -e "${BLUE}════════════════════════════════════════${NC}"
  printf "${BLUE}  课次  %-6s  %-8s  %-8s${NC}\n" "页数" "section" "代码引用"
  echo -e "${BLUE}────────────────────────────────────────${NC}"
  local total=0
  local total_src=0
  for f in slides.01.md slides.02.md slides.03.md slides.04.md \
           slides.05.md slides.06.md slides.07.md slides.08.md; do
    local n="${f:7:2}"
    # slides = layout: 声明数
    local pages
    pages=$(grep -c "^layout:" "$f" 2>/dev/null || echo 0)
    local divs
    divs=$(grep -c "^layout: section" "$f" 2>/dev/null || echo 0)
    local refs
    refs=$(grep -c '<SourceCode' "$f" 2>/dev/null || echo 0)
    printf "  L%02d   %-6s  %-8s  %-8s\n" "$((10#$n))" "$pages" "$divs" "$refs"
    total=$((total + pages))
    total_src=$((total_src + refs))
  done
  echo -e "${BLUE}────────────────────────────────────────${NC}"
  printf "${GREEN}  合计  %-6s  %-8s  %-8s${NC}\n" "$total" "-" "$total_src"
  echo -e "${BLUE}════════════════════════════════════════${NC}"
}

# ── Lint 检查 ─────────────────────────────────────────
lint() {
  local issues=0
  echo -e "${CYAN}▶ 检查 HTML 标签平衡...${NC}"

  for f in slides.0{1,2,3,4,5,6,7,8}.md; do
    # 检查 div 标签
    local open_div close_div
    open_div=$(grep -c '<div' "$f" 2>/dev/null || echo 0)
    close_div=$(grep -c '</div>' "$f" 2>/dev/null || echo 0)
    if [[ "$open_div" -ne "$close_div" ]]; then
      echo -e "  ${RED}✗ $f: <div> 不匹配 (开:$open_div 闭:$close_div)${NC}"
      issues=$((issues + 1))
    fi
    # 检查 code 标签
    local open_code close_code
    open_code=$(grep -c '<code>' "$f" 2>/dev/null || echo 0)
    close_code=$(grep -c '</code>' "$f" 2>/dev/null || echo 0)
    if [[ "$open_code" -ne "$close_code" ]]; then
      echo -e "  ${YELLOW}⚠ $f: <code> 不匹配 (开:$open_code 闭:$close_code)${NC}"
      issues=$((issues + 1))
    fi
    # 检查 strong 标签
    local open_s close_s
    open_s=$(grep -c '<strong>' "$f" 2>/dev/null || echo 0)
    close_s=$(grep -c '</strong>' "$f" 2>/dev/null || echo 0)
    if [[ "$open_s" -ne "$close_s" ]]; then
      echo -e "  ${YELLOW}⚠ $f: <strong> 不匹配 (开:$open_s 闭:$close_s)${NC}"
      issues=$((issues + 1))
    fi
  done

  echo -e "${CYAN}▶ 检查 SelfTest ID 唯一性...${NC}"
  local dupes
  dupes=$(grep -roh 'id="l[0-9][0-9]-q[0-9]"' slides.0{1,2,3,4,5,6,7,8}.md 2>/dev/null | sort | uniq -d)
  if [[ -n "$dupes" ]]; then
    echo -e "  ${RED}✗ 重复的 SelfTest ID:${NC}"
    echo "$dupes"
    issues=$((issues + 1))
  fi

  echo -e "${CYAN}▶ 检查 <pre> 标签（可能导致 Vue 解析错误）...${NC}"
  for f in slides.0{1,2,3,4,5,6,7,8}.md; do
    if grep -q '<pre>' "$f" 2>/dev/null; then
      echo -e "  ${YELLOW}⚠ $f: 包含 <pre> 标签，检查是否在 SelfTest answer 属性内${NC}"
      issues=$((issues + 1))
    fi
  done

  echo -e "${CYAN}▶ 检查多余闭合标签...${NC}"
  for f in slides.0{1,2,3,4,5,6,7,8}.md; do
    # 检查 --- 分隔符后紧跟 </div>（slide 边界处多余的闭合标签）
    local stray
    stray=$(grep -A1 '^---$' "$f" 2>/dev/null | grep -c '^</div>$' || true)
    if [[ "$stray" -gt 0 ]]; then
      echo -e "  ${YELLOW}⚠ $f: --- 后紧接 </div>（$stray 处）${NC}"
      issues=$((issues + 1))
    fi
    # 检查连续两个 </div>（可能是重复闭合，也可能是正常的相邻 div 关闭）
    local dup
    dup=$(grep -n '^</div>$' "$f" 2>/dev/null | awk -F: 'NR>1{d=$1-prev; prev=$1; if(d==1)print "line "($1-1)" and " $1}' || true)
    if [[ -n "$dup" ]]; then
      echo -e "  ${BLUE}ℹ $f: 连续 </div> (通常正常) 在 $dup${NC}"
    fi
  done

  if [[ "$issues" -eq 0 ]]; then
    echo -e "${GREEN}✓ 未发现问题${NC}"
  else
    echo -e "${RED}✗ 发现 $issues 个问题${NC}"
  fi
}

# ── 检查源码引用 ──────────────────────────────────────
check_refs() {
  echo -e "${CYAN}▶ 校验 SourceCode 引用...${NC}"
  local repo_root
  repo_root="$(cd "$SCRIPT_DIR/../../.." && pwd)"
  local total=0 errors=0

  for f in slides.0{1,2,3,4,5,6,7,8}.md; do
    while IFS= read -r line; do
      # 提取 file="..." 和 lines="..."
      local file
      file=$(echo "$line" | sed -n 's/.*file="\([^"]*\)".*/\1/p' | head -1)
      local lns
      lns=$(echo "$line" | sed -n 's/.*lines="\([^"]*\)".*/\1/p' | head -1)
      [[ -z "$file" ]] && continue

      total=$((total + 1))
      local full="${repo_root}/${file}"

      if [[ ! -f "$full" ]]; then
        echo -e "  ${RED}✗ $f: 文件不存在 — $file${NC}"
        errors=$((errors + 1))
        continue
      fi

      if [[ -n "$lns" ]]; then
        local max_line
        max_line=$(wc -l < "$full" | tr -d ' ')
        local start
        start=$(echo "$lns" | cut -d'-' -f1)
        if [[ "$start" -gt "$max_line" ]]; then
          echo -e "  ${RED}✗ $f: 行号超界 — $file:$lns (文件只有 $max_line 行)${NC}"
          errors=$((errors + 1))
        fi
      fi
    done < <(grep '<SourceCode' "$f" 2>/dev/null || true)
  done

  echo -e "${GREEN}✓ 校验 $total 个引用，$errors 个错误${NC}"
}

# ── 信息 ──────────────────────────────────────────────
info() {
  local version
  version=$(npx slidev --version 2>/dev/null || echo "未安装")
  echo -e "${BLUE}nano-vllm 课件 项目概览${NC}"
  echo -e "──────────────────────────────────"
  echo -e "Slidev 版本:  ${CYAN}$version${NC}"
  echo -e "课程数:      ${CYAN}8${NC}"
  echo -e "总页数:      ${CYAN}$(grep -c '^layout:' slides.0{1,2,3,4,5,6,7,8}.md 2>/dev/null | awk -F: '{s+=$2} END{print s}')${NC}"
  echo -e "自定义组件:  ${CYAN}3${NC} (SelfTest, SourceCode, ProgressWidget)"
  echo -e "入口文件:    ${CYAN}slides.md${NC}"
  echo -e "主题:        ${CYAN}@slidev/theme-default${NC}"
  echo
  echo -e "快速命令:"
  echo -e "  bash manage.sh dev        ${CYAN}# 启动开发服务器${NC}"
  echo -e "  bash manage.sh build      ${CYAN}# 构建到 dist/${NC}"
  echo -e "  bash manage.sh lint       ${CYAN}# 检查常见问题${NC}"
}

# ── 主入口 ────────────────────────────────────────────
case "${1:-help}" in
  dev)       shift; dev "$@" ;;
  build)     build ;;
  serve)     serve "${2:-}" ;;
  clean)     clean "${2:-}" ;;
  lint)      lint ;;
  count)     count ;;
  check-refs) check_refs ;;
  info)      info ;;
  help|--help|-h) help ;;
  *)
    echo -e "${RED}未知命令: $1${NC}"
    echo "运行 bash manage.sh help 查看帮助"
    exit 1
    ;;
esac
