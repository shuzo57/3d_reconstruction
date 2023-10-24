import argparse
import os

import cv2
from ultralytics import YOLO  # type: ignore

from .config import SAVE_VIDEO_EXT
from .utils import get_basename, get_video_paths


def plot_skeleton_to_video(
    input, output_dir, pose_model_path, club_model_path
):
    pose_model = YOLO(pose_model_path)
    club_model = YOLO(club_model_path)

    for video_path in get_video_paths(input):
        print(f"Processing {video_path}")
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_name = get_basename(video_path)

        output_path = os.path.join(output_dir, video_name + SAVE_VIDEO_EXT)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        while True:
            ret, frame = cap.read()
            if ret:
                print(
                    f"\r {cap.get(cv2.CAP_PROP_POS_FRAMES)} / "
                    + f"{cap.get(cv2.CAP_PROP_FRAME_COUNT)}",
                    end="",
                )

                pose_results = pose_model(frame, verbose=False)
                club_results = club_model(frame, verbose=False)
                pose_frame = pose_results[0].plot()
                club_frame = club_results[0].plot()

                pose_frame = cv2.putText(
                    pose_frame,
                    str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                video_frame = cv2.hconcat([pose_frame, club_frame])
                out.write(video_frame)
            else:
                print("\n")
                break

        cap.release()
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--pose_model_path", type=str, required=True)
    parser.add_argument("--club_model_path", type=str, required=True)
    args = parser.parse_args()

    plot_skeleton_to_video(
        args.video_path,
        args.output_path,
        args.pose_model_path,
        args.club_model_path,
    )
