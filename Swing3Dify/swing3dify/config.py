video_extensions = [".MP4", ".mp4", ".avi", ".mkv", ".MOV", ".mov"]

POSE_DIR_NAME = "pose"
CLUB_DIR_NAME = "club"
IMG_DIR_NAME = "img"
DATA_DIR_NAME = "data"
RECONSTRUCTED_DIR_NAME = "data_3d"
FIGURE_DIR_NAME = "figure"
FIGURE_EXT = ".png"

POSITION_FILE_NAME = "position_data.csv"
CONFIDENCE_FILE_NAME = "confidence_data.csv"
RECONSTRUCTED_FILE_NAME = "3d_position_data.csv"

SAVE_VIDEO_EXT = ".mp4"

TARGET_CLASS = 0
CONF_THR = 0.90
DEFAULT_IMG_HEIGHT = 1920
DEFAULT_IMG_WIDTH = 1080

CLUB_POSITION_COLUMNS = [
    "frame",
    "BOX_x",
    "BOX_y",
    "BOX_width",
    "BOX_height",
    "TOE_x",
    "TOE_y",
    "HOSEL_x",
    "HOSEL_y",
    "GRIP_x",
    "GRIP_y",
]

CLUB_CONF_COLUMNS = [
    "frame",
    "BOX_conf",
    "TOE_conf",
    "HOSEL_conf",
    "GRIP_conf",
]

POSE_POSITION_COLUMNS = [
    "frame",
    "NOSE_x",
    "NOSE_y",
    "LEFT_EYE_x",
    "LEFT_EYE_y",
    "RIGHT_EYE_x",
    "RIGHT_EYE_y",
    "LEFT_EAR_x",
    "LEFT_EAR_y",
    "RIGHT_EAR_x",
    "RIGHT_EAR_y",
    "LEFT_SHOULDER_x",
    "LEFT_SHOULDER_y",
    "RIGHT_SHOULDER_x",
    "RIGHT_SHOULDER_y",
    "LEFT_ELBOW_x",
    "LEFT_ELBOW_y",
    "RIGHT_ELBOW_x",
    "RIGHT_ELBOW_y",
    "LEFT_WRIST_x",
    "LEFT_WRIST_y",
    "RIGHT_WRIST_x",
    "RIGHT_WRIST_y",
    "LEFT_HIP_x",
    "LEFT_HIP_y",
    "RIGHT_HIP_x",
    "RIGHT_HIP_y",
    "LEFT_KNEE_x",
    "LEFT_KNEE_y",
    "RIGHT_KNEE_x",
    "RIGHT_KNEE_y",
    "LEFT_ANKLE_x",
    "LEFT_ANKLE_y",
    "RIGHT_ANKLE_x",
    "RIGHT_ANKLE_y",
]

POSE_CONF_COLUMNS = [
    "frame",
    "NOSE_conf",
    "LEFT_EYE_conf",
    "RIGHT_EYE_conf",
    "LEFT_EAR_conf",
    "RIGHT_EAR_conf",
    "LEFT_SHOULDER_conf",
    "RIGHT_SHOULDER_conf",
    "LEFT_ELBOW_conf",
    "RIGHT_ELBOW_conf",
    "LEFT_WRIST_conf",
    "RIGHT_WRIST_conf",
    "LEFT_HIP_conf",
    "RIGHT_HIP_conf",
    "LEFT_KNEE_conf",
    "RIGHT_KNEE_conf",
    "LEFT_ANKLE_conf",
    "RIGHT_ANKLE_conf",
]

POSE_KEYPOINTS = [
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

ALL_KEYPOINTS = [
    "NOSE",
    "LEFT_EYE",
    "RIGHT_EYE",
    "LEFT_EAR",
    "RIGHT_EAR",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "TOE",
    "HOSEL",
    "GRIP",
]
