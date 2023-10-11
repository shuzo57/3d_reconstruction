import numpy as np
import pandas as pd

from .calibration_data import KNOWN_CALIBRATION_DATA
from .config import ALL_KEYPOINTS, CONF_THR
from .time_adjustment import calculate_delay_frame
from .utils import compute_camera_parameters, reconstruct_3D


def get_synced_data(
    club1: pd.DataFrame,
    club2: pd.DataFrame,
    conf1: pd.DataFrame,
    conf2: pd.DataFrame,
    part_name: str = "HOSEL",
):
    club1.interpolate(method="linear", both=True, inplace=True)
    club2.interpolate(method="linear", both=True, inplace=True)

    delay_frame = calculate_delay_frame(club1, club2, part_name)

    new_club1 = club1.copy()
    new_club2 = club2.copy()

    for c in new_club1.columns:
        if c.endswith("_x") or c.endswith("_y"):
            new_club1.loc[conf1["BOX_conf"] <= CONF_THR, c] = None
            new_club2.loc[conf2["BOX_conf"] <= CONF_THR, c] = None

    new_club1["new_frame"] = new_club1["frame"]
    new_club2["new_frame"] = new_club2["frame"] + delay_frame

    index1 = set(
        new_club1[~new_club1.isnull().any(axis=1)]["new_frame"].values
    )
    index2 = set(
        new_club2[~new_club2.isnull().any(axis=1)]["new_frame"].values
    )

    common_index = np.array(sorted(list(index1 & index2)))

    new_club1 = new_club1[new_club1["new_frame"].isin(common_index)].copy()
    new_club2 = new_club2[new_club2["new_frame"].isin(common_index)].copy()

    pts1 = np.float32(new_club1[[f"{part_name}_x", f"{part_name}_y"]].values)
    pts2 = np.float32(new_club2[[f"{part_name}_x", f"{part_name}_y"]].values)

    return pts1, pts2


def synced_data_to_camera_parameters(
    pts1: np.ndarray, pts2: np.ndarray, camera_type: str = "FDR-AX700"
):
    K = np.array(KNOWN_CALIBRATION_DATA[camera_type]["mtx"])

    R, T, F = compute_camera_parameters(pts1, pts2, K)  # type: ignore
    return R, T, F, K


def generate_reconstructed_3d_data(
    club1: pd.DataFrame,
    club2: pd.DataFrame,
    pose1: pd.DataFrame,
    pose2: pd.DataFrame,
    K: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    part_name: str = "HOSEL",
):
    club1.interpolate(method="linear", both=True, inplace=True)
    club2.interpolate(method="linear", both=True, inplace=True)

    drop_columns = ["BOX_x", "BOX_y", "BOX_width", "BOX_height"]
    club1.drop(columns=drop_columns, inplace=True)
    club2.drop(columns=drop_columns, inplace=True)

    pose1.interpolate(method="linear", both=True, inplace=True)
    pose2.interpolate(method="linear", both=True, inplace=True)

    delay_frame = calculate_delay_frame(club1, club2, part_name)

    club1_min = club1.index.min()
    club1_max = club1.index.max()
    club2_min = club2.index.min() + delay_frame
    club2_max = club2.index.max() + delay_frame

    min_index = max(club1_min, club2_min)
    max_index = min(club1_max, club2_max)

    df1 = pose1.merge(club1, on="frame")
    df2 = pose2.merge(club2, on="frame")

    df1 = df1.iloc[min_index:max_index].copy()
    df2 = df2.iloc[
        min_index - delay_frame : max_index - delay_frame  # noqa
    ].copy()

    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    keypoint_labels = np.array(
        [[f"{k}_x", f"{k}_y", f"{k}_z"] for k in ALL_KEYPOINTS]
    ).flatten()
    df = pd.DataFrame(columns=keypoint_labels)

    for frame in df1.index:
        pts1 = np.float32(df1.iloc[frame, 1:].values).reshape((-1, 2))
        pts2 = np.float32(df2.iloc[frame, 1:].values).reshape((-1, 2))

        reconstructed_3d = reconstruct_3D(pts1, pts2, K, R, T)
        df.loc[len(df)] = reconstructed_3d.T.flatten()
    df.insert(0, "frame", range(min_index, max_index))
    return df
