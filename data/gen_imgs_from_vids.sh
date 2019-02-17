#!/bin/bash

if [ -z "$1" ]
then
    echo "Provide the path to videos as a wildcard like: gen_imgs_from_vids own/videos/*.mp4"
else
    wildcard="$1"
fi

j=0
for i in $wildcard; do
    [ -f "$i" ] || break

    ffmpeg -i "$i" -r 10 -s 160x120 -f image2 own/table3/imgs/table-$j-%05d.jpg
    j=$((j+1))

done
