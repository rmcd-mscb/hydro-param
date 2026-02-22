#!/bin/bash
# Block direct edits to lock files — regenerate with pixi install instead
INPUT=$(cat)
FILE_PATH=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['tool_input']['file_path'])" "$INPUT" 2>/dev/null)

BASENAME=$(basename "$FILE_PATH")

if [[ "$BASENAME" == "pixi.lock" || "$BASENAME" == "package-lock.json" ]]; then
  echo '{"hookSpecificOutput":{"decision":"block","reason":"Never edit lock files directly. Run pixi install to regenerate pixi.lock."}}'
  exit 2
fi

exit 0
