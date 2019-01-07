#!/bin/bash

wildcard="own/table3/videos/*.mp4"

j=0
for i in $wildcard; do
    [ -f "$i" ] || break

    ffmpeg -i "$i" -r 10 -s 160x120 -f image2 own/table3/imgs/table-$j-%05d.jpg
    j=$((j+1))

done
