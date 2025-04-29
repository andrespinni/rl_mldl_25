#!/usr/bin/env bash

window=100
input="rewards.txt"
movfile="movavg.txt"

awk -v w=$window '
{
  a[NR] = $1
  sum += $1
  if (NR > w) {
    sum -= a[NR-w]
  }
  if (NR >= w) {
    # stampo: episodio, media mobile
    printf "%d %.6f\n", NR, sum/w
  }
}
' "$input" > "$movfile"
