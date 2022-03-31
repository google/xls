#!/bin/sh

if test -n "$(diff -w -q "$1" "$2")"; then
  {
    printf "[ERROR] %s and %s differ\n\n" "$1" "$2"
    printf "Diff:\n\n"
    diff -w "$1" "$2"
  } 1>&2
  exit 1
fi
exit 0
