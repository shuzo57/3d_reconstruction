import cv2
import numpy as np


def reconstruct_3D(pts1, pts2, K, R, T):
    P1 = K.dot(np.hstack([np.eye(3), np.zeros((3, 1))]))
    P2 = K.dot(np.hstack([R, T.reshape(3, -1)]))

    homogeneous_3D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    reconstructed_3D = homogeneous_3D[:3] / homogeneous_3D[3]
    return reconstructed_3D
