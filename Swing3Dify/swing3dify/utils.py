import os
import pandas as pd

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
