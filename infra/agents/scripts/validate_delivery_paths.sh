#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <repo-path> [<repo-path> ...]" >&2
  exit 2
fi

status=0

for path in "$@"; do
  if [ -e "$path" ]; then
    echo "OK  $path"
  else
    echo "MISSING  $path" >&2
    status=1
  fi
done

exit "$status"
