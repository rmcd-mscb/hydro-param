#!/bin/bash
# Run mypy on edited Python files for immediate type-error feedback
INPUT=$(cat)
FILE_PATH=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['tool_input']['file_path'])" "$INPUT" 2>/dev/null)

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
