import logging
import os
import sys

import cv2
import pandas as pd

from .config import (
    CLUB_DIR_NAME,
    CONFIDENCE_FILE_NAME,
    DATA_DIR_NAME,
    FIGURE_DIR_NAME,
    FIGURE_EXT,
    POSE_DIR_NAME,
    POSITION_FILE_NAME,
    RECONSTRUCTED_DIR_NAME,
    RECONSTRUCTED_FILE_NAME,
)
from .core import (
    generate_reconstructed_3d_data,
    get_synced_data,
    synced_data_to_camera_parameters,
)
from .utils import get_basename, rescale_data
from .visualizations import (
    draw_epipolar_lines,
    draw_feature_matches,
    show_3d_swing,
)
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
    _, img1 = cap1.read()
    _, img2 = cap2.read()
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
    pts1, pts2 = get_synced_data(club1, club2, conf1, conf2)
    print(f"Number of feature matches: {len(pts1)}")
    R, T, F, K = synced_data_to_camera_parameters(pts1, pts2)

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

    logging.info("Save figures")
    save_dir1 = os.path.join(
        output_path, DATA_DIR_NAME, video_name1, FIGURE_DIR_NAME
    )
    os.makedirs(save_dir1, exist_ok=True)
    save_dir2 = os.path.join(
        output_path, DATA_DIR_NAME, video_name2, FIGURE_DIR_NAME
    )
    os.makedirs(save_dir2, exist_ok=True)

    print("save: feature matches")
    save_path1 = os.path.join(
        save_dir1, f"draw_feature_matches_{video_name1}{FIGURE_EXT}"
    )
    save_path2 = os.path.join(
        save_dir2, f"draw_feature_matches_{video_name2}{FIGURE_EXT}"
    )
    draw_feature_matches(img1, img2, pts1, pts2, SAVE_PATH=save_path1)
    draw_feature_matches(img2, img1, pts2, pts1, SAVE_PATH=save_path2)

    print("save: epipolar lines")
    save_path1 = os.path.join(
        save_dir1, f"draw_epipolar_lines_{video_name1}{FIGURE_EXT}"
    )
    save_path2 = os.path.join(
        save_dir2, f"draw_epipolar_lines_{video_name2}{FIGURE_EXT}"
    )
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    draw_epipolar_lines(
        gray_img1, gray_img2, pts1, pts2, F, SAVE_PATH=save_path1
    )
    draw_epipolar_lines(
        gray_img1, gray_img2, pts1, pts2, F, SAVE_PATH=save_path2
    )

    print("save: 3D Swing animation")
    save_path1 = os.path.join(
        save_dir1, f"3d_reconstruction_{video_name1}.html"
    )
    save_path2 = os.path.join(
        save_dir2, f"3d_reconstruction_{video_name2}.html"
    )
    show_3d_swing(
        reconstructed_3d_df, window=10, frame_step=10, SAVE_PATH=save_path1
    )
    show_3d_swing(
        reconstructed_3d_df, window=10, frame_step=10, SAVE_PATH=save_path2
    )

    logging.info("Done")
