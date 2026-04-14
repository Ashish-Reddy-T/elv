#!/bin/bash
# PostToolUse hook: run ruff check + format after Write/Edit on Python files

FILE=$(jq -r '.tool_input.file_path' < /dev/stdin)

if [[ ! "$FILE" =~ \.py$ ]]; then
  exit 0
fi

if [[ ! -f "$FILE" ]]; then
  exit 0
fi

ruff check --fix "$FILE" 2>/dev/null
ruff format "$FILE" 2>/dev/null

ERRORS=$(ruff check "$FILE" 2>&1)
if [[ -n "$ERRORS" ]]; then
  echo "$ERRORS" >&2
  exit 1
fi

exit 0
