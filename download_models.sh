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

download_from_drive "1G1YQkpQmPK9LXy6CJkTOjgeAs7dE9-mX" "club-s.pt"
download_from_drive "1BJSpv0SU_od0WAs0VHCSJ-ydJpUKzgyY" "pose-l.pt"