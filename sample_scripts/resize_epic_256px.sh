# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash
indir="./EPIC-KITCHENS/epic/train"
outdir="./EPIC-KITCHENS/train_256/"

cd $indir
videos=$(find . -iname *.MP4)

num_procs=64  # Run this many in parallel at max
num_jobs="\j"  # The prompt escape for number of jobs currently running
for video in $videos; do
    while (( ${num_jobs@P} >= num_procs )); do
        wait -n
    done
    mkdir -p $(dirname ${outdir}/${video})
    # from https://superuser.com/a/624564
    # ffmpeg -y -i ${indir}/${video} -filter:v scale="trunc(oh*a/2)*2:256" -c:a copy ${outdir}/${video} &
    ffmpeg -y -i ${indir}/${video} -filter:v scale="trunc(oh*a/2)*2:256" ${outdir}/${video} &
    echo 'Converted ' ${video}
done
