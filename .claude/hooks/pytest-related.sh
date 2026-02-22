#!/bin/bash
# Run related test file after editing a Python source module
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

# Extract module name from file path
BASENAME=$(basename "$FILE_PATH" .py)

# Map subpackage modules to their test names
case "$FILE_PATH" in
  */derivations/pywatershed.py)
    TEST_NAME="test_pywatershed_derivation" ;;
  */formatters/pywatershed.py)
    TEST_NAME="test_pywatershed_formatter" ;;
  *)
    TEST_NAME="test_${BASENAME}" ;;
esac

PROJECT_DIR=$(echo "$FILE_PATH" | sed 's|/src/hydro_param/.*||')
TEST_FILE="${PROJECT_DIR}/tests/${TEST_NAME}.py"

# Only run if a matching test file exists
if [[ ! -f "$TEST_FILE" ]]; then
  exit 0
fi

pixi run -e dev pytest "$TEST_FILE" -x -q 2>&1
exit 0
