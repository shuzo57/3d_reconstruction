import logging
import os
import sys

import cv2
import pandas as pd

from .config import (
    CLUB_DIR_NAME,
    CONFIDENCE_FILE_NAME,
    DATA_DIR_NAME,
    POSE_DIR_NAME,
    POSITION_FILE_NAME,
    RECONSTRUCTED_DIR_NAME,
    RECONSTRUCTED_FILE_NAME,
)
from .core import (
    dataframe_to_camera_parameters,
    generate_reconstructed_3d_data,
)
from .utils import get_basename, rescale_data
from .YoloClub import YoloClub
from .YoloPose import YoloPose


def run(
    file_path1: str,
    file_path2: str,
    output_path: str,
    pose_model_path: str,
    club_model_path: str,
    save_images: bool,
):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.info(f"Input: {file_path1}, {file_path2}")

    logging.info("Start pose estimation")
    YoloPose(file_path1, output_path, pose_model_path, save_images)
    YoloPose(file_path2, output_path, pose_model_path, save_images)

    logging.info("Start club position estimation")
    YoloClub(file_path1, output_path, club_model_path, save_images)
    YoloClub(file_path2, output_path, club_model_path, save_images)

    logging.info("Start 3D reconstruction")
    logging.info("Load data")
    video_name1 = get_basename(file_path1)
    video_name2 = get_basename(file_path2)

    pose1_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name1,
        POSE_DIR_NAME,
        POSITION_FILE_NAME,
    )
    pose2_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name2,
        POSE_DIR_NAME,
        POSITION_FILE_NAME,
    )
    club1_position_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name1,
        CLUB_DIR_NAME,
        POSITION_FILE_NAME,
    )
    club2_position_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name2,
        CLUB_DIR_NAME,
        POSITION_FILE_NAME,
    )
    club1_confidence_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name1,
        CLUB_DIR_NAME,
        CONFIDENCE_FILE_NAME,
    )
    club2_confidence_path = os.path.join(
        output_path,
        DATA_DIR_NAME,
        video_name2,
        CLUB_DIR_NAME,
        CONFIDENCE_FILE_NAME,
    )
    pose1 = pd.read_csv(pose1_path)
    pose2 = pd.read_csv(pose2_path)
    club1 = pd.read_csv(club1_position_path)
    club2 = pd.read_csv(club2_position_path)
    conf1 = pd.read_csv(club1_confidence_path)
    conf2 = pd.read_csv(club2_confidence_path)

    logging.info("Check the size of the input videos")
    cap1 = cv2.VideoCapture(file_path1)
    cap2 = cv2.VideoCapture(file_path2)
    img_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap1.release()
    cap2.release()

    if img_width1 != img_width2 or img_height1 != img_height2:
        logging.error("The size of the input videos must be the same.")
        sys.exit(1)
    else:
        img_width = img_width1
        img_height = img_height1

        logging.info("Rescale data")
        pose1 = rescale_data(pose1, img_width, img_height)
        pose2 = rescale_data(pose2, img_width, img_height)
        club1 = rescale_data(club1, img_width, img_height)
        club2 = rescale_data(club2, img_width, img_height)

    logging.info("Compute camera parameters")
    R, T, _, K = dataframe_to_camera_parameters(club1, club2, conf1, conf2)

    logging.info("Reconstruct 3D data")
    reconstructed_3d_df = generate_reconstructed_3d_data(
        club1, club2, pose1, pose2, K, R, T
    )

    logging.info("Save reconstructed 3D data")
    save_dir1 = os.path.join(
        output_path, DATA_DIR_NAME, video_name1, RECONSTRUCTED_DIR_NAME
    )
    os.makedirs(save_dir1, exist_ok=True)
    save_dir2 = os.path.join(
        output_path, DATA_DIR_NAME, video_name2, RECONSTRUCTED_DIR_NAME
    )
    os.makedirs(save_dir2, exist_ok=True)
    save_path1 = os.path.join(save_dir1, RECONSTRUCTED_FILE_NAME)
    save_path2 = os.path.join(save_dir2, RECONSTRUCTED_FILE_NAME)
    reconstructed_3d_df.to_csv(save_path1, index=False, header=True)
    reconstructed_3d_df.to_csv(save_path2, index=False, header=True)

    logging.info("Done")
