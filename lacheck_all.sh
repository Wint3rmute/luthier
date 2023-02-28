#!/bin/bash

# Runs lacheck on you .tex files, exits with error 1
# if lacheck reports any errors with your tex spells

set -Eeuo pipefail
cd "$(dirname "$(readlink -f "$0")")"/latex

for file in *.tex; do
  if [[ "$file" = "Dyplom.tex" ]]; then
    continue # Don't fix what's broken by PWr :)
  fi

  printf "Checking $file.. "

  if [ $(lacheck "$file" | wc -l) != 0 ]; then
    echo "$file has some problems!"
    lacheck "$file" 
    exit 1
  fi

  echo "OK"
done
