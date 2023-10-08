#!/bin/bash

DOWNLOAD_FOLDER="models"

mkdir -p "${DOWNLOAD_FOLDER}"

download_from_drive() {
    local FILE_ID="$1"
    local FILE_NAME="$2"

    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    local CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o "${DOWNLOAD_FOLDER}/${FILE_NAME}"
}

download_from_drive "1U3dSjXypKJ7YjyKcjjDKPC_nDjJED89m" "club-v1.pt" # baseline model
download_from_drive "1BJSpv0SU_od0WAs0VHCSJ-ydJpUKzgyY" "pose-l.pt"
download_from_drive "1dL1tfWogB9s6rfj5Oe6Hx4CWGtUKNqlU" "club-yokota.pt" # for yokota club house