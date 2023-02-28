#!/usr/bin/bash
set -Eeuo pipefail

# Fixes broken windows-specific encoding formats
# used for polish characters, converts all to utf-8

for file in $(fd -e tex -e bib);
do
  echo "Fixing $file.."
  iconv -f WINDOWS-1250 -t utf-8 "$file" > /tmp/out.tex
  cp /tmp/out.tex "$file"
done

echo
echo "All done!"
