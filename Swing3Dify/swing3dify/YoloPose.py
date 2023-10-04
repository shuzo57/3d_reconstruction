import argparse
import logging
import os

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .config import (
    CONFIDENCE_FILE_NAME,
    DATA_DIR_NAME,
    IMG_DIR_NAME,
    POSE_CONF_COLUMNS,
    POSE_DIR_NAME,
    POSE_POSITION_COLUMNS,
    POSITION_FILE_NAME,
)
from .utils import get_basename, get_video_paths


def YoloPose(input, output, model_path, save_images) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    logging.info(f"Input: {input}")
    if save_images:
        img_dir = os.path.join(output, IMG_DIR_NAME)
        os.makedirs(img_dir, exist_ok=True)
    data_dir = os.path.join(output, DATA_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)

    model = YOLO(model_path)
    print(model.info(verbose=True))

    for video_path in get_video_paths(input):
        logging.info(f"Processing {video_path}")
        video_name = get_basename(video_path)

        if save_images:
            video_img_dir = os.path.join(img_dir, video_name, POSE_DIR_NAME)
            os.makedirs(video_img_dir, exist_ok=True)
        video_data_dir = os.path.join(data_dir, video_name, POSE_DIR_NAME)
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
                    os.path.join(video_data_dir, POSITION_FILE_NAME),
                    index=False,
                    header=True,
                )
                conf_df.to_csv(
                    os.path.join(video_data_dir, CONFIDENCE_FILE_NAME),
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
        "-p", "--path", help="path to model", default="yolov8s-pose.pt"
    )
    parser.add_argument(
        "-s", "--save_images", help="save images", type=bool, default=False
    )
    args = parser.parse_args()
    YoloPose(args.input, args.output, args.path, args.save_images)
