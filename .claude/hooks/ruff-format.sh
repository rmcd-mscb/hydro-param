#!/bin/bash
# Auto-format Python files after Edit/Write using ruff
INPUT=$(cat)
FILE_PATH=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['tool_input']['file_path'])" "$INPUT" 2>/dev/null)

# Only run on Python files
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Only run if file exists
if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

pixi run -e dev ruff format "$FILE_PATH" 2>/dev/null
pixi run -e dev ruff check --fix "$FILE_PATH" 2>/dev/null
exit 0
