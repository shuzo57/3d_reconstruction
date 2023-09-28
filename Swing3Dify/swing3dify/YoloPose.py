import argparse
import logging
import os

import cv2
import numpy as np
import pandas as pd
from .config import POSE_CONF_COLUMNS, POSE_POSITION_COLUMNS
from ultralytics import YOLO
from .utils import get_video_paths


def YoloPose(input, output, model_path, rotate_direction, save_images) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    logging.info(f"Input: {input}")
    if save_images:
        img_dir = os.path.join(output, "img")
        os.makedirs(img_dir, exist_ok=True)
    data_dir = os.path.join(output, "data")
    os.makedirs(data_dir, exist_ok=True)

    model = YOLO(model_path)
    print(model.info(verbose=True))

    for video_path in get_video_paths(input):
        logging.info(f"Processing {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        if save_images:
            video_img_dir = os.path.join(img_dir, video_name)
            os.makedirs(video_img_dir, exist_ok=True)
        video_data_dir = os.path.join(data_dir, video_name)
        os.makedirs(video_data_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        position_df = pd.DataFrame(columns=POSE_POSITION_COLUMNS)
        conf_df = pd.DataFrame(columns=POSE_CONF_COLUMNS)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if ret:
                frame_idx += 1
                print(f"\r frame: {frame_idx}", end="")

                if rotate_direction == "right":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotate_direction == "left":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                results = model(frame, verbose=False)

                if save_images:
                    plotted_frame = results[0].plot()
                    cv2.imwrite(
                        os.path.join(
                            video_img_dir, f"{video_name}_{frame_idx}.jpg"
                        ),
                        plotted_frame,
                    )

                boxes = results[0].cpu().numpy().boxes
                keypoints = results[0].cpu().numpy().keypoints

                if len(boxes.conf) > 0:
                    idx = np.argmax(boxes.conf)

                    key_data = keypoints.xyn[idx].flatten()
                    key_conf = keypoints.conf[idx]

                    key_data = np.insert(key_data, 0, frame_idx)
                    key_conf = np.insert(key_conf, 0, frame_idx)

                    position_df.loc[len(position_df)] = [*key_data]
                    conf_df.loc[len(conf_df)] = [*key_conf]
                else:
                    position_df.loc[len(position_df)] = [
                        np.nan if _ != 0 else frame_idx
                        for _ in range(len(POSE_POSITION_COLUMNS))
                    ]
                    conf_df.loc[len(conf_df)] = [
                        np.nan if _ != 0 else frame_idx
                        for _ in range(len(POSE_CONF_COLUMNS))
                    ]
            else:
                print("")
                position_df.to_csv(
                    os.path.join(video_data_dir, "position_data.csv"),
                    index=False,
                    header=True,
                )
                conf_df.to_csv(
                    os.path.join(video_data_dir, "confidence_data.csv"),
                    index=False,
                    header=True,
                )
                logging.info(f"Output CSV files for {video_name}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="input video file or directory", required=True
    )
    parser.add_argument(
        "-o", "--output", help="output directory", default=os.getcwd()
    )
    parser.add_argument(
        "-p", "--path", help="path to model", default="yolov5s.pt"  # 後で修正
    )
    parser.add_argument(
        "-r",
        "--rotate",
        type=str,
        choices=["right", "left", "none"],
        default="none",
        help="Rotation direction: 'right' for clockwise, 'left' for "
        + "counterclockwise, 'none' for no rotation (default)",
    )
    parser.add_argument(
        "-s", "--save_images", help="save images", type=bool, default=False
    )
    args = parser.parse_args()
    YoloPose(args.input, args.output, args.path, args.rotate, args.save_images)
