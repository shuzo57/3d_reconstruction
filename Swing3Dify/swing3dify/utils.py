import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def draw_feature_matches(img1, img2, pts1, pts2, figsize=(10, 10)):
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

    scatter = go.Scatter3d(x=x, y=y, z=z, mode="markers")

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(nticks=10, range=[np.min(x), np.max(x)]),
            yaxis=dict(nticks=10, range=[np.min(y), np.max(y)]),
            zaxis=dict(nticks=10, range=[np.min(z), np.max(z)]),
        ),
    )

    fig = go.Figure(data=[scatter], layout=layout)
    fig.show()
