import cv2
import numpy as np


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
    F, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    E = K.T.dot(F).dot(K)

    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = U.dot(W).dot(Vt)
    T = U[:, 2]

    return R, T
