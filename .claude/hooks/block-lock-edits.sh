#!/bin/bash
# Block direct edits to lock files — regenerate with pixi install instead
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['tool_input']['file_path'])" 2>&1)
PARSE_STATUS=$?

if [[ $PARSE_STATUS -ne 0 || -z "$FILE_PATH" ]]; then
  echo '{"hookSpecificOutput":{"decision":"block","reason":"Hook error: could not parse file path from tool input. Blocking as a precaution."}}'
  exit 2
fi

BASENAME=$(basename "$FILE_PATH")

if [[ "$BASENAME" == "pixi.lock" ]]; then
  echo '{"hookSpecificOutput":{"decision":"block","reason":"Never edit pixi.lock directly. Run pixi install to regenerate it."}}'
  exit 2
fi

exit 0
