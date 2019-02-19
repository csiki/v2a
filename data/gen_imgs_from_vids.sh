#!/bin/bash

if [ -z "$1" ]
then
    echo "Provide the path to the output images like: ./gen_imgs_from_vids.sh out/path/ own/videos/*.mp4"
    exit 2
else
    outpath="$1"
fi

if [ -z "$2" ]
then
    echo "Provide the path to videos as a wildcard like: ./gen_imgs_from_vids.sh out/path/ own/videos/*.mp4"
    exit 1
fi

fps=10  # change fps here if needed

j=0
for i in "${@:2}"; do

    [ -f "$i" ] || break
    echo "Processing: $i"

    ffmpeg -i "$i" -r $fps -s 160x120 -f image2 $outpath/$j-%05d.jpg
    j=$((j+1))

done
