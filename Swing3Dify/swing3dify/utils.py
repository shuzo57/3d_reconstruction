import os

import cv2
import numpy as np
import pandas as pd

from .config import video_extensions


def reconstruct_3D(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
):
    P1 = K.dot(np.hstack([np.eye(3), np.zeros((3, 1))]))
    P2 = K.dot(np.hstack([R, T.reshape(3, -1)]))

    homogeneous_3D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    reconstructed_3D = homogeneous_3D[:3] / homogeneous_3D[3]
    return reconstructed_3D


def compute_camera_parameters(
    pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray
):
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    E = K.T.dot(F).dot(K)

    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U.dot(W).dot(Vt)
    T = U[:, 2]

    return R, T, F


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


def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]


def calculate_body_part_center(df: pd.DataFrame, target: str):
    for axis in ["x", "y", "z"]:
        if target != "MOUTH":
            df[f"CENTER_{target}_{axis}"] = df[
                [f"LEFT_{target}_{axis}", f"RIGHT_{target}_{axis}"]
            ].mean(axis="columns")
        else:
            df[f"CENTER_{target}_{axis}"] = df[
                [f"{target}_LEFT_{axis}", f"{target}_RIGHT_{axis}"]
            ].mean(axis="columns")
    return df


def rescale_data(df, width, height):
    for c in df.columns:
        if c.endswith("_x") or c.endswith("_width"):
            df[c] = df[c] * width
        elif c.endswith("_y") or c.endswith("_height"):
            df[c] = df[c] * height
    return df


def scale_data_to_real_distance(
    df, start_point, end_point, real_world_distance
):
    """
    Scale the data such that the distance between the start and end points
    in the first frame matches the given real world distance.

    Parameters:
    - df: DataFrame containing the 3D coordinates
    - start_point: Name of the start body point (e.g., "LEFT_EYE")
    - end_point: Name of the end body point (e.g., "RIGHT_EYE")
    - real_world_distance: The actual distance between the start
    and end points in the real world

    Returns:
    - DataFrame with scaled 3D coordinates
    """
    # Compute the distance in the first frame
    start_coords = np.array(
        [
            df[start_point + "_x"].iloc[0],
            df[start_point + "_y"].iloc[0],
            df[start_point + "_z"].iloc[0],
        ]
    )
    end_coords = np.array(
        [
            df[end_point + "_x"].iloc[0],
            df[end_point + "_y"].iloc[0],
            df[end_point + "_z"].iloc[0],
        ]
    )
    current_distance = np.linalg.norm(start_coords - end_coords)

    # Compute the scaling factor
    scaling_factor = real_world_distance / current_distance

    # Apply the scaling factor to the entire dataframe
    scaled_df = df.copy()
    scaled_df.iloc[:, 1:] = scaled_df.iloc[:, 1:] * scaling_factor

    return scaled_df


def get_body_vectors(df: pd.DataFrame, frame):
    target_dict = {}

    EYE_x = df[["LEFT_EYE_x", "RIGHT_EYE_x"]].loc[frame]
    EYE_y = df[["LEFT_EYE_y", "RIGHT_EYE_y"]].loc[frame]
    EYE_z = df[["LEFT_EYE_z", "RIGHT_EYE_z"]].loc[frame]
    target_dict["EYE_x"] = EYE_x
    target_dict["EYE_y"] = EYE_y
    target_dict["EYE_z"] = EYE_z

    SHOULDER_x = df[["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]].loc[frame]
    SHOULDER_y = df[["LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]].loc[frame]
    SHOULDER_z = df[["LEFT_SHOULDER_z", "RIGHT_SHOULDER_z"]].loc[frame]
    target_dict["SHOULDER_x"] = SHOULDER_x
    target_dict["SHOULDER_y"] = SHOULDER_y
    target_dict["SHOULDER_z"] = SHOULDER_z

    SHOULDER_x = df[["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]].loc[frame]
    SHOULDER_y = df[["LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]].loc[frame]
    SHOULDER_z = df[["LEFT_SHOULDER_z", "RIGHT_SHOULDER_z"]].loc[frame]
    target_dict["SHOULDER_x"] = SHOULDER_x
    target_dict["SHOULDER_y"] = SHOULDER_y
    target_dict["SHOULDER_z"] = SHOULDER_z

    LEFT_UPPER_ARM_x = df[["LEFT_SHOULDER_x", "LEFT_ELBOW_x"]].loc[frame]
    LEFT_UPPER_ARM_y = df[["LEFT_SHOULDER_y", "LEFT_ELBOW_y"]].loc[frame]
    LEFT_UPPER_ARM_z = df[["LEFT_SHOULDER_z", "LEFT_ELBOW_z"]].loc[frame]
    target_dict["LEFT_UPPER_ARM_x"] = LEFT_UPPER_ARM_x
    target_dict["LEFT_UPPER_ARM_y"] = LEFT_UPPER_ARM_y
    target_dict["LEFT_UPPER_ARM_z"] = LEFT_UPPER_ARM_z

    RIGHT_UPPER_ARM_x = df[["RIGHT_SHOULDER_x", "RIGHT_ELBOW_x"]].loc[frame]
    RIGHT_UPPER_ARM_y = df[["RIGHT_SHOULDER_y", "RIGHT_ELBOW_y"]].loc[frame]
    RIGHT_UPPER_ARM_z = df[["RIGHT_SHOULDER_z", "RIGHT_ELBOW_z"]].loc[frame]
    target_dict["RIGHT_UPPER_ARM_x"] = RIGHT_UPPER_ARM_x
    target_dict["RIGHT_UPPER_ARM_y"] = RIGHT_UPPER_ARM_y
    target_dict["RIGHT_UPPER_ARM_z"] = RIGHT_UPPER_ARM_z

    LEFT_FOREARM_x = df[["LEFT_ELBOW_x", "LEFT_WRIST_x"]].loc[frame]
    LEFT_FOREARM_y = df[["LEFT_ELBOW_y", "LEFT_WRIST_y"]].loc[frame]
    LEFT_FOREARM_z = df[["LEFT_ELBOW_z", "LEFT_WRIST_z"]].loc[frame]
    target_dict["LEFT_FOREARM_x"] = LEFT_FOREARM_x
    target_dict["LEFT_FOREARM_y"] = LEFT_FOREARM_y
    target_dict["LEFT_FOREARM_z"] = LEFT_FOREARM_z

    RIGHT_FOREARM_x = df[["RIGHT_ELBOW_x", "RIGHT_WRIST_x"]].loc[frame]
    RIGHT_FOREARM_y = df[["RIGHT_ELBOW_y", "RIGHT_WRIST_y"]].loc[frame]
    RIGHT_FOREARM_z = df[["RIGHT_ELBOW_z", "RIGHT_WRIST_z"]].loc[frame]
    target_dict["RIGHT_FOREARM_x"] = RIGHT_FOREARM_x
    target_dict["RIGHT_FOREARM_y"] = RIGHT_FOREARM_y
    target_dict["RIGHT_FOREARM_z"] = RIGHT_FOREARM_z

    HIP_x = df[["LEFT_HIP_x", "RIGHT_HIP_x"]].loc[frame]
    HIP_y = df[["LEFT_HIP_y", "RIGHT_HIP_y"]].loc[frame]
    HIP_z = df[["LEFT_HIP_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["HIP_x"] = HIP_x
    target_dict["HIP_y"] = HIP_y
    target_dict["HIP_z"] = HIP_z

    LEFT_BODY_x = df[["LEFT_SHOULDER_x", "LEFT_HIP_x"]].loc[frame]
    LEFT_BODY_y = df[["LEFT_SHOULDER_y", "LEFT_HIP_y"]].loc[frame]
    LEFT_BODY_z = df[["LEFT_SHOULDER_z", "LEFT_HIP_z"]].loc[frame]
    target_dict["LEFT_BODY_x"] = LEFT_BODY_x
    target_dict["LEFT_BODY_y"] = LEFT_BODY_y
    target_dict["LEFT_BODY_z"] = LEFT_BODY_z

    RIGHT_BODY_x = df[["RIGHT_SHOULDER_x", "RIGHT_HIP_x"]].loc[frame]
    RIGHT_BODY_y = df[["RIGHT_SHOULDER_y", "RIGHT_HIP_y"]].loc[frame]
    RIGHT_BODY_z = df[["RIGHT_SHOULDER_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["RIGHT_BODY_x"] = RIGHT_BODY_x
    target_dict["RIGHT_BODY_y"] = RIGHT_BODY_y
    target_dict["RIGHT_BODY_z"] = RIGHT_BODY_z

    LEFT_THIGH_x = df[["LEFT_HIP_x", "LEFT_KNEE_x"]].loc[frame]
    LEFT_THIGH_y = df[["LEFT_HIP_y", "LEFT_KNEE_y"]].loc[frame]
    LEFT_THIGH_z = df[["LEFT_HIP_z", "LEFT_KNEE_z"]].loc[frame]
    target_dict["LEFT_THIGH_x"] = LEFT_THIGH_x
    target_dict["LEFT_THIGH_y"] = LEFT_THIGH_y
    target_dict["LEFT_THIGH_z"] = LEFT_THIGH_z

    RIGHT_THIGH_x = df[["RIGHT_HIP_x", "RIGHT_KNEE_x"]].loc[frame]
    RIGHT_THIGH_y = df[["RIGHT_HIP_y", "RIGHT_KNEE_y"]].loc[frame]
    RIGHT_THIGH_z = df[["RIGHT_HIP_z", "RIGHT_KNEE_z"]].loc[frame]
    target_dict["RIGHT_THIGH_x"] = RIGHT_THIGH_x
    target_dict["RIGHT_THIGH_y"] = RIGHT_THIGH_y
    target_dict["RIGHT_THIGH_z"] = RIGHT_THIGH_z

    LEFT_LOWER_LEG_x = df[["LEFT_KNEE_x", "LEFT_ANKLE_x"]].loc[frame]
    LEFT_LOWER_LEG_y = df[["LEFT_KNEE_y", "LEFT_ANKLE_y"]].loc[frame]
    LEFT_LOWER_LEG_z = df[["LEFT_KNEE_z", "LEFT_ANKLE_z"]].loc[frame]
    target_dict["LEFT_LOWER_LEG_x"] = LEFT_LOWER_LEG_x
    target_dict["LEFT_LOWER_LEG_y"] = LEFT_LOWER_LEG_y
    target_dict["LEFT_LOWER_LEG_z"] = LEFT_LOWER_LEG_z

    RIGHT_LOWER_LEG_x = df[["RIGHT_KNEE_x", "RIGHT_ANKLE_x"]].loc[frame]
    RIGHT_LOWER_LEG_y = df[["RIGHT_KNEE_y", "RIGHT_ANKLE_y"]].loc[frame]
    RIGHT_LOWER_LEG_z = df[["RIGHT_KNEE_z", "RIGHT_ANKLE_z"]].loc[frame]
    target_dict["RIGHT_LOWER_LEG_x"] = RIGHT_LOWER_LEG_x
    target_dict["RIGHT_LOWER_LEG_y"] = RIGHT_LOWER_LEG_y
    target_dict["RIGHT_LOWER_LEG_z"] = RIGHT_LOWER_LEG_z

    return target_dict


def get_swing_vectors(df: pd.DataFrame, frame):
    target_dict = {}
    ClUB_x = df[["GRIP_x", "HOSEL_x"]].loc[frame]
    CLUB_y = df[["GRIP_y", "HOSEL_y"]].loc[frame]
    ClUB_z = df[["GRIP_z", "HOSEL_z"]].loc[frame]
    target_dict["CLUB_x"] = ClUB_x
    target_dict["CLUB_y"] = CLUB_y
    target_dict["CLUB_z"] = ClUB_z

    EYE_x = df[["LEFT_EYE_x", "RIGHT_EYE_x"]].loc[frame]
    EYE_y = df[["LEFT_EYE_y", "RIGHT_EYE_y"]].loc[frame]
    EYE_z = df[["LEFT_EYE_z", "RIGHT_EYE_z"]].loc[frame]
    target_dict["EYE_x"] = EYE_x
    target_dict["EYE_y"] = EYE_y
    target_dict["EYE_z"] = EYE_z

    SHOULDER_x = df[["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]].loc[frame]
    SHOULDER_y = df[["LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]].loc[frame]
    SHOULDER_z = df[["LEFT_SHOULDER_z", "RIGHT_SHOULDER_z"]].loc[frame]
    target_dict["SHOULDER_x"] = SHOULDER_x
    target_dict["SHOULDER_y"] = SHOULDER_y
    target_dict["SHOULDER_z"] = SHOULDER_z

    SHOULDER_x = df[["LEFT_SHOULDER_x", "RIGHT_SHOULDER_x"]].loc[frame]
    SHOULDER_y = df[["LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]].loc[frame]
    SHOULDER_z = df[["LEFT_SHOULDER_z", "RIGHT_SHOULDER_z"]].loc[frame]
    target_dict["SHOULDER_x"] = SHOULDER_x
    target_dict["SHOULDER_y"] = SHOULDER_y
    target_dict["SHOULDER_z"] = SHOULDER_z

    LEFT_UPPER_ARM_x = df[["LEFT_SHOULDER_x", "LEFT_ELBOW_x"]].loc[frame]
    LEFT_UPPER_ARM_y = df[["LEFT_SHOULDER_y", "LEFT_ELBOW_y"]].loc[frame]
    LEFT_UPPER_ARM_z = df[["LEFT_SHOULDER_z", "LEFT_ELBOW_z"]].loc[frame]
    target_dict["LEFT_UPPER_ARM_x"] = LEFT_UPPER_ARM_x
    target_dict["LEFT_UPPER_ARM_y"] = LEFT_UPPER_ARM_y
    target_dict["LEFT_UPPER_ARM_z"] = LEFT_UPPER_ARM_z

    RIGHT_UPPER_ARM_x = df[["RIGHT_SHOULDER_x", "RIGHT_ELBOW_x"]].loc[frame]
    RIGHT_UPPER_ARM_y = df[["RIGHT_SHOULDER_y", "RIGHT_ELBOW_y"]].loc[frame]
    RIGHT_UPPER_ARM_z = df[["RIGHT_SHOULDER_z", "RIGHT_ELBOW_z"]].loc[frame]
    target_dict["RIGHT_UPPER_ARM_x"] = RIGHT_UPPER_ARM_x
    target_dict["RIGHT_UPPER_ARM_y"] = RIGHT_UPPER_ARM_y
    target_dict["RIGHT_UPPER_ARM_z"] = RIGHT_UPPER_ARM_z

    LEFT_FOREARM_x = df[["LEFT_ELBOW_x", "LEFT_WRIST_x"]].loc[frame]
    LEFT_FOREARM_y = df[["LEFT_ELBOW_y", "LEFT_WRIST_y"]].loc[frame]
    LEFT_FOREARM_z = df[["LEFT_ELBOW_z", "LEFT_WRIST_z"]].loc[frame]
    target_dict["LEFT_FOREARM_x"] = LEFT_FOREARM_x
    target_dict["LEFT_FOREARM_y"] = LEFT_FOREARM_y
    target_dict["LEFT_FOREARM_z"] = LEFT_FOREARM_z

    RIGHT_FOREARM_x = df[["RIGHT_ELBOW_x", "RIGHT_WRIST_x"]].loc[frame]
    RIGHT_FOREARM_y = df[["RIGHT_ELBOW_y", "RIGHT_WRIST_y"]].loc[frame]
    RIGHT_FOREARM_z = df[["RIGHT_ELBOW_z", "RIGHT_WRIST_z"]].loc[frame]
    target_dict["RIGHT_FOREARM_x"] = RIGHT_FOREARM_x
    target_dict["RIGHT_FOREARM_y"] = RIGHT_FOREARM_y
    target_dict["RIGHT_FOREARM_z"] = RIGHT_FOREARM_z

    HIP_x = df[["LEFT_HIP_x", "RIGHT_HIP_x"]].loc[frame]
    HIP_y = df[["LEFT_HIP_y", "RIGHT_HIP_y"]].loc[frame]
    HIP_z = df[["LEFT_HIP_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["HIP_x"] = HIP_x
    target_dict["HIP_y"] = HIP_y
    target_dict["HIP_z"] = HIP_z

    LEFT_BODY_x = df[["LEFT_SHOULDER_x", "LEFT_HIP_x"]].loc[frame]
    LEFT_BODY_y = df[["LEFT_SHOULDER_y", "LEFT_HIP_y"]].loc[frame]
    LEFT_BODY_z = df[["LEFT_SHOULDER_z", "LEFT_HIP_z"]].loc[frame]
    target_dict["LEFT_BODY_x"] = LEFT_BODY_x
    target_dict["LEFT_BODY_y"] = LEFT_BODY_y
    target_dict["LEFT_BODY_z"] = LEFT_BODY_z

    RIGHT_BODY_x = df[["RIGHT_SHOULDER_x", "RIGHT_HIP_x"]].loc[frame]
    RIGHT_BODY_y = df[["RIGHT_SHOULDER_y", "RIGHT_HIP_y"]].loc[frame]
    RIGHT_BODY_z = df[["RIGHT_SHOULDER_z", "RIGHT_HIP_z"]].loc[frame]
    target_dict["RIGHT_BODY_x"] = RIGHT_BODY_x
    target_dict["RIGHT_BODY_y"] = RIGHT_BODY_y
    target_dict["RIGHT_BODY_z"] = RIGHT_BODY_z

    LEFT_THIGH_x = df[["LEFT_HIP_x", "LEFT_KNEE_x"]].loc[frame]
    LEFT_THIGH_y = df[["LEFT_HIP_y", "LEFT_KNEE_y"]].loc[frame]
    LEFT_THIGH_z = df[["LEFT_HIP_z", "LEFT_KNEE_z"]].loc[frame]
    target_dict["LEFT_THIGH_x"] = LEFT_THIGH_x
    target_dict["LEFT_THIGH_y"] = LEFT_THIGH_y
    target_dict["LEFT_THIGH_z"] = LEFT_THIGH_z

    RIGHT_THIGH_x = df[["RIGHT_HIP_x", "RIGHT_KNEE_x"]].loc[frame]
    RIGHT_THIGH_y = df[["RIGHT_HIP_y", "RIGHT_KNEE_y"]].loc[frame]
    RIGHT_THIGH_z = df[["RIGHT_HIP_z", "RIGHT_KNEE_z"]].loc[frame]
    target_dict["RIGHT_THIGH_x"] = RIGHT_THIGH_x
    target_dict["RIGHT_THIGH_y"] = RIGHT_THIGH_y
    target_dict["RIGHT_THIGH_z"] = RIGHT_THIGH_z

    LEFT_LOWER_LEG_x = df[["LEFT_KNEE_x", "LEFT_ANKLE_x"]].loc[frame]
    LEFT_LOWER_LEG_y = df[["LEFT_KNEE_y", "LEFT_ANKLE_y"]].loc[frame]
    LEFT_LOWER_LEG_z = df[["LEFT_KNEE_z", "LEFT_ANKLE_z"]].loc[frame]
    target_dict["LEFT_LOWER_LEG_x"] = LEFT_LOWER_LEG_x
    target_dict["LEFT_LOWER_LEG_y"] = LEFT_LOWER_LEG_y
    target_dict["LEFT_LOWER_LEG_z"] = LEFT_LOWER_LEG_z

    RIGHT_LOWER_LEG_x = df[["RIGHT_KNEE_x", "RIGHT_ANKLE_x"]].loc[frame]
    RIGHT_LOWER_LEG_y = df[["RIGHT_KNEE_y", "RIGHT_ANKLE_y"]].loc[frame]
    RIGHT_LOWER_LEG_z = df[["RIGHT_KNEE_z", "RIGHT_ANKLE_z"]].loc[frame]
    target_dict["RIGHT_LOWER_LEG_x"] = RIGHT_LOWER_LEG_x
    target_dict["RIGHT_LOWER_LEG_y"] = RIGHT_LOWER_LEG_y
    target_dict["RIGHT_LOWER_LEG_z"] = RIGHT_LOWER_LEG_z

    return target_dict
