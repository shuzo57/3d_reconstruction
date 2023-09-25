import os

from .config import video_extensions


def get_video_paths(path):
    video_paths = []

    if os.path.isfile(path):
        if any(path.endswith(ext) for ext in video_extensions):
            video_paths.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(root, file))
    return video_paths
