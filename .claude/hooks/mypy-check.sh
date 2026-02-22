#!/bin/bash
# Run mypy on edited Python files for immediate type-error feedback
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['tool_input']['file_path'])" 2>&1)
PARSE_STATUS=$?

if [[ $PARSE_STATUS -ne 0 || -z "$FILE_PATH" ]]; then
  echo "WARNING: mypy-check.sh could not parse file path from hook input. Skipping." >&2
  exit 0
fi

# Only run on Python files under src/hydro_param/
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi
case "$FILE_PATH" in
  */src/hydro_param/*) ;;
  *) exit 0 ;;
esac

# Only run if file exists
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

pixi run -e dev mypy "$FILE_PATH" 2>&1
exit 0
