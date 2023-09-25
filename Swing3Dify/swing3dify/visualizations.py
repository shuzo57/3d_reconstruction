import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import plotly.graph_objects as go


def draw_feature_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    RATE_FALG: bool = False,
    figsize: tuple = (10, 10),
):
    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1, _ = rgb_img1.shape
    h2, w2, _ = rgb_img2.shape

    height = min(h1, h2)
    width = w1 + w2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:h1, :w1, :] = rgb_img1[:height, :width, :]
    img[:h2, w1:, :] = rgb_img2[:height, :width, :]

    cmap = plt.get_cmap("hsv")

    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if RATE_FALG:
            pt1 = pt1 * np.array([w1, h1])
            pt2 = pt2 * np.array([w2, h2])
        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int) + np.array([w1, 0]))

        color = cmap(i / len(pts1))[:-1]
        color = tuple([int(c * 255) for c in color])

        cv2.circle(img, pt1, 5, color, -1)
        cv2.circle(img, pt2, 5, color, -1)
        cv2.line(img, pt1, pt2, color, 1)

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    plt.show()


def visualize_3d_points(points_3d):
    x, y, z = points_3d

    colormap = plt.get_cmap("hsv")
    colors = np.linspace(0, 1, len(x))
    colors = colormap(colors)

    scatter = go.Scatter3d(
        x=x, y=y, z=z, mode="markers", marker=dict(color=colors)
    )

    max_range = max(np.ptp(x), np.ptp(y), np.ptp(z))

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                nticks=10,
                range=[np.mean(x) - max_range / 2, np.mean(x) + max_range / 2],
            ),
            yaxis=dict(
                nticks=10,
                range=[np.mean(y) - max_range / 2, np.mean(y) + max_range / 2],
            ),
            zaxis=dict(
                nticks=10,
                range=[np.mean(z) - max_range / 2, np.mean(z) + max_range / 2],
            ),
        ),
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()


def plot_3d_points(points_3d):
    x, y, z = points_3d

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colormap = plt.get_cmap("hsv")
    colors = np.linspace(0, 1, len(x))
    colors = colormap(colors)

    ax.scatter(x, y, z, c=colors, marker="o")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def drawlines(
    img1: np.ndarray,
    img2: np.ndarray,
    lines: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
):
    """img1, img2: gray images"""
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(map(int, pt1)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2)), 5, color, -1)
    return img1, img2


def draw_epipolar_lines(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    figsize: tuple = (10, 10),
):
    """img1, img2: gray images"""
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, _ = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, _ = drawlines(img2, img1, lines2, pts2, pts1)

    h1, w1, _ = img5.shape
    h2, w2, _ = img3.shape

    height = min(h1, h2)
    width = w1 + w2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:h1, :w1, :] = img5[:height, :width, :]
    img[:h2, w1:, :] = img3[:height, :width, :]

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis("off")
    plt.show()
