#!/bin/bash
# Auto-format Python files after Edit/Write using ruff
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['tool_input']['file_path'])" 2>&1)
PARSE_STATUS=$?

if [[ $PARSE_STATUS -ne 0 || -z "$FILE_PATH" ]]; then
  echo "WARNING: ruff-format.sh could not parse file path from hook input. Skipping." >&2
  exit 0
fi

# Only run on Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Only run if file exists (Write could have been to a new path)
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

RUFF_FMT_OUTPUT=$(pixi run -e dev ruff format "$FILE_PATH" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "ruff format failed for $FILE_PATH:" >&2
  echo "$RUFF_FMT_OUTPUT" >&2
fi

RUFF_CHK_OUTPUT=$(pixi run -e dev ruff check --fix "$FILE_PATH" 2>&1)
if [[ $? -ne 0 ]]; then
  echo "ruff check --fix failed for $FILE_PATH:" >&2
  echo "$RUFF_CHK_OUTPUT" >&2
fi

exit 0
