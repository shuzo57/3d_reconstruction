import argparse
import os
import sys

from .run import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D reconstruction from two video files."
    )

    parser.add_argument("-f1", "--file1", help="Path to the first video file.")
    parser.add_argument(
        "-f2", "--file2", help="Path to the second video file."
    )
    parser.add_argument("-o", "--output", help="Path to the output directory.")
    parser.add_argument(
        "-p", "--pose_model", help="Path to the pose estimation model."
    )
    parser.add_argument(
        "-c",
        "--club_model",
        help="Path to the club position estimation model.",
    )
    parser.add_argument(
        "-s", "--save_images", help="save images", type=bool, default=False
    )
    args = parser.parse_args()

    if not os.path.exists(args.file1):
        print(f"File {args.file1} does not exist.")
        sys.exit(1)
    if not os.path.exists(args.file2):
        print(f"File {args.file2} does not exist.")
        sys.exit(1)

    run(
        args.file1,
        args.file2,
        args.output,
        args.pose_model,
        args.club_model,
        args.save_images,
    )
