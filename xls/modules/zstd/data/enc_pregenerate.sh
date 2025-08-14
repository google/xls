#!/bin/bash

DATA_DIR="xls/modules/zstd/data"
sizes=(1024 1030 200 300 500 2000)

for size in "${sizes[@]}";do
    filename=enc_pregenerated_"$size"B
    filepath="$DATA_DIR/$filename"
    dd if=/dev/urandom of="$filepath" bs=1B count="$size"
done
