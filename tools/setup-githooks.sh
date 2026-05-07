#!/usr/bin/env bash
# =============================================
# tools/setup-githooks.sh
# 역할: 본 repo (또는 분리 repo) 의 git hook 활성화.
#       - core.hooksPath = .githooks
#       - .githooks/* 실행 권한 부여
#
# 사용 (현재 repo):
#   bash tools/setup-githooks.sh
#
# 사용 (분리 repo 에 동일 hook 배포):
#   bash tools/setup-githooks.sh /path/to/AeroInspect_backend
#   bash tools/setup-githooks.sh /path/to/AeroInspect_frontend
#   → 통합 repo 의 .githooks/* 를 대상 repo 에 복사 + 동일 활성화
# =============================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC_HOOKS="$HERE/.githooks"
TARGET="${1:-$HERE}"

if [ ! -d "$SRC_HOOKS" ]; then
  echo "❌ source hooks 디렉터리를 찾을 수 없습니다: $SRC_HOOKS" >&2
  exit 1
fi
if [ ! -d "$TARGET/.git" ]; then
  echo "❌ git repo 가 아닙니다: $TARGET" >&2
  exit 1
fi

# 1) 분리 repo 라면 hook 복사
if [ "$TARGET" != "$HERE" ]; then
  mkdir -p "$TARGET/.githooks"
  cp "$SRC_HOOKS"/pre-commit "$TARGET/.githooks/"
  cp "$SRC_HOOKS"/commit-msg "$TARGET/.githooks/"
  echo "[setup] hooks 복사: $TARGET/.githooks/"
fi

# 2) 실행 권한
chmod +x "$TARGET/.githooks/pre-commit" "$TARGET/.githooks/commit-msg" 2>/dev/null || true

# 3) core.hooksPath 설정
git -C "$TARGET" config core.hooksPath .githooks
echo "[setup] core.hooksPath = .githooks  ($TARGET)"

# 4) 검증
echo "[setup] 활성 hook:"
git -C "$TARGET" config --get core.hooksPath
ls -la "$TARGET/.githooks" | grep -v '^total' | grep -v '^\.$' || true
echo ""
echo "✅ git hook 활성화 완료."
echo "   pre-commit  : Vibe_Coding_Log.md 갱신 강제"
echo "   commit-msg  : Conventional Commits 형식 강제"
echo ""
echo "💡 우회가 필요한 경우 (드물게만):"
echo "   SKIP_VIBE_LOG_CHECK=1 git commit ..."
echo "   SKIP_COMMIT_MSG_CHECK=1 git commit ..."
