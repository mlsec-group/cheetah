#!/usr/bin/env bash

set -eu

JOERN_PARSE="/opt/joern/joern-cli/joern-parse"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input-path> <output-path>"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
TMP_ROOT=$(mktemp -d)
trap 'rm -rf -- "$TMP_ROOT"' EXIT

# hotfixes for joern-parse (C++ files)
# 1.  remove static keyword and namespace from input if present
# 2.  remove namespace from input if present
# 3.  remove template parameters from input if present
# 4 remove additional function modifiers (e.g. override)
sed '1s/^static //' "$INPUT_PATH" \
    | sed 's/[a-zA-Z0-9_]*:://g' \
    | sed 's/<[a-zA-Z0-9_]*>//g' \
    | sed '1s/\((.*)\)[a-zA-Z0-9_ ]*\({\)/\1 \2/' \
    | sed '1s/\((.*)\)[a-zA-Z0-9_ ]*$/\1/' \
    | sed '1s/&/*/g' \
 > "$TMP_ROOT/$(basename $INPUT_PATH)"
$JOERN_PARSE "$TMP_ROOT/$(basename $INPUT_PATH)" -o "$OUTPUT_PATH"
