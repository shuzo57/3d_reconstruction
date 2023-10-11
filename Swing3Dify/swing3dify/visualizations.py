import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from .utils import get_body_vectors, get_swing_vectors


def draw_feature_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    RATE_FALG: bool = False,
    figsize: tuple = (10, 10),
    SAVE_PATH: str = "",
):
    rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h1, w1, _ = rgb_img1.shape
    h2, w2, _ = rgb_img2.shape

    height = max(h1, h2)
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

    if SAVE_PATH != "":
        plt.savefig(SAVE_PATH)
        plt.close()
    else:
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

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    plt.show()


def drawlines(
    img1: np.ndarray,
    img2: np.ndarray,
    lines: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])  # type: ignore
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])  # type: ignore
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
    SAVE_PATH: str = "",
):
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

    if SAVE_PATH != "":
        plt.savefig(SAVE_PATH)
        plt.close()
    else:
        plt.show()


def show_3d_human_pose(
    df: pd.DataFrame,
    frame: int,
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    tick_interval: float = 0.05,
    SAVE_PATH: str = "",
) -> None:
    vec_data = get_body_vectors(df, frame)

    x_vec_label = list(vec_data.keys())[0::3]
    y_vec_label = list(vec_data.keys())[1::3]
    z_vec_label = list(vec_data.keys())[2::3]
    vec_name = [label.replace("_x", "") for label in x_vec_label]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data[x_label],
                y=vec_data[y_label],
                z=vec_data[z_label],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                name=name,
            )
            for x_label, y_label, z_label, name in zip(
                x_vec_label, y_vec_label, z_vec_label, vec_name
            )
        ]
    )
    min_x_value = min(df.filter(like="_x").loc[frame])
    max_x_value = max(df.filter(like="_x").loc[frame])
    min_y_value = min(df.filter(like="_y").loc[frame])
    max_y_value = max(df.filter(like="_y").loc[frame])
    min_z_value = min(df.filter(like="_z").loc[frame])
    max_z_value = max(df.filter(like="_z").loc[frame])

    min_x_value = int(min_x_value / tick_interval) * tick_interval
    max_x_value = int(max_x_value / tick_interval) * tick_interval
    min_y_value = int(min_y_value / tick_interval) * tick_interval
    max_y_value = int(max_y_value / tick_interval) * tick_interval
    min_z_value = int(min_z_value / tick_interval) * tick_interval
    max_z_value = int(max_z_value / tick_interval) * tick_interval

    scene = dict(
        xaxis=dict(
            tickvals=np.arange(
                min_x_value - tick_interval,
                max_x_value + tick_interval,
                tick_interval,
            )
        ),
        yaxis=dict(
            tickvals=np.arange(
                min_y_value - tick_interval,
                max_y_value + tick_interval,
                tick_interval,
            )
        ),
        zaxis=dict(
            tickvals=np.arange(
                min_z_value - tick_interval,
                max_z_value + tick_interval,
                tick_interval,
            )
        ),
    )

    fig.update_layout(scene=scene)

    if SAVE_PATH != "":
        fig.write_html(SAVE_PATH)
    else:
        fig.show()


def show_3d_swing_pose(
    df: pd.DataFrame,
    frame: int,
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    tick_interval: float = 0.05,
    SAVE_PATH: str = "",
) -> None:
    vec_data = get_swing_vectors(df, frame)

    x_vec_label = list(vec_data.keys())[0::3]
    y_vec_label = list(vec_data.keys())[1::3]
    z_vec_label = list(vec_data.keys())[2::3]
    vec_name = [label.replace("_x", "") for label in x_vec_label]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data[x_label],
                y=vec_data[y_label],
                z=vec_data[z_label],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                name=name,
            )
            for x_label, y_label, z_label, name in zip(
                x_vec_label, y_vec_label, z_vec_label, vec_name
            )
        ]
    )

    min_x_value = min(df.filter(like="_x").loc[frame])
    max_x_value = max(df.filter(like="_x").loc[frame])
    min_y_value = min(df.filter(like="_y").loc[frame])
    max_y_value = max(df.filter(like="_y").loc[frame])
    min_z_value = min(df.filter(like="_z").loc[frame])
    max_z_value = max(df.filter(like="_z").loc[frame])

    min_x_value = int(min_x_value / tick_interval) * tick_interval
    max_x_value = int(max_x_value / tick_interval) * tick_interval
    min_y_value = int(min_y_value / tick_interval) * tick_interval
    max_y_value = int(max_y_value / tick_interval) * tick_interval
    min_z_value = int(min_z_value / tick_interval) * tick_interval
    max_z_value = int(max_z_value / tick_interval) * tick_interval

    scene = dict(
        xaxis=dict(
            tickvals=np.arange(
                min_x_value - tick_interval,
                max_x_value + tick_interval,
                tick_interval,
            )
        ),
        yaxis=dict(
            tickvals=np.arange(
                min_y_value - tick_interval,
                max_y_value + tick_interval,
                tick_interval,
            )
        ),
        zaxis=dict(
            tickvals=np.arange(
                min_z_value - tick_interval,
                max_z_value + tick_interval,
                tick_interval,
            )
        ),
    )

    fig.update_layout(scene=scene)

    if SAVE_PATH != "":
        fig.write_html(SAVE_PATH)
    else:
        fig.show()


def show_3d_swing(
    df: pd.DataFrame,
    line_width: int = 10,
    marker_size: int = 3,
    graph_mode: str = "lines+markers",
    window: int = 1,
    frame_step: int = 1,
    SAVE_PATH: str = "",
) -> None:
    if window is None:
        df = df.rolling(window, center=True).mean()
        df = df.dropna()

    x_max = df.filter(like="_x").max().max()
    x_min = df.filter(like="_x").min().min()
    y_max = df.filter(like="_y").max().max()
    y_min = df.filter(like="_y").min().min()
    z_max = df.filter(like="_z").max().max()
    z_min = df.filter(like="_z").min().min()

    min_frame = df.index.min()
    max_frame = df.index.max()

    frames = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        vec_data = get_swing_vectors(df, frame)

        x_vec_label = list(vec_data.keys())[0::3]
        y_vec_label = list(vec_data.keys())[1::3]
        z_vec_label = list(vec_data.keys())[2::3]
        vec_name = [label.replace("_x", "") for label in x_vec_label]

        fig = go.Frame(
            data=[
                go.Scatter3d(
                    x=vec_data[x_label],
                    y=vec_data[y_label],
                    z=vec_data[z_label],
                    mode=graph_mode,
                    line=dict(width=line_width),
                    marker=dict(size=marker_size),
                    name=name,
                )
                for x_label, y_label, z_label, name in zip(
                    x_vec_label, y_vec_label, z_vec_label, vec_name
                )
            ],
            name=f"{frame}",
            layout=go.Layout(title=f"frame:{frame}"),
        )
        frames.append(fig)

    vec_data = get_swing_vectors(df, min_frame)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vec_data[x_label],
                y=vec_data[y_label],
                z=vec_data[z_label],
                mode=graph_mode,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                name=name,
            )
            for x_label, y_label, z_label, name in zip(
                x_vec_label, y_vec_label, z_vec_label, vec_name
            )
        ],
        frames=frames,
    )

    steps = []
    for frame in range(min_frame, max_frame + 1, frame_step):
        step = dict(
            method="animate",
            args=[
                [f"{frame}"],
                dict(frame=dict(duration=1, redraw=True), mode="immediate"),
            ],
            label=f"{frame}",
        )
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
            active=0,
            transition=dict(duration=0),
            currentvalue=dict(
                font=dict(size=20), prefix="", visible=True, xanchor="right"
            ),
        )
    ]

    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1.0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0.5, z=-1.5),
            ),
            xaxis=dict(title="x-axis", range=[x_min, x_max]),
            yaxis=dict(title="y-axis", range=[y_min, y_max]),
            zaxis=dict(title="z-axis", range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.0),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                xanchor="left",
                yanchor="top",
                x=0,
                y=1,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=sliders,
    )
    if SAVE_PATH != "":
        fig.write_html(
            SAVE_PATH,
            auto_play=False,
        )
    else:
        fig.show()
