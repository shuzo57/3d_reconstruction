#!/bin/bash

DOWNLOAD_FOLDER="videos"

mkdir -p "${DOWNLOAD_FOLDER}"

download_from_drive() {
    local FILE_ID="$1"
    local FILE_NAME="$2"

    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
    local CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o "${DOWNLOAD_FOLDER}/${FILE_NAME}"
}

download_from_drive "1f1Ar5o0lhN1gp-iRlrUVcVRE3EwHmqjD" "sample_video1.mp4"
download_from_drive "104SZsrpxQ0UF2TTUjyQ_34G1kEfHsdVb" "sample_video2.mp4"