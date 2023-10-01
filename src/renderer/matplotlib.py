# From TEMOS: temos/render/anim.py
# Assume Z is the gravity axis
# Inspired by
# - https://github.com/anindita127/Complextext2animation/blob/main/src/utils/visualization.py
# - https://github.com/facebookresearch/QuaterNet/blob/main/common/visualization.py

import logging

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from src.rifke import canonicalize_rotation

logger = logging.getLogger("matplotlib.animation")
logger.setLevel(logging.ERROR)

colors = ("black", "magenta", "red", "green", "blue")

KINEMATIC_TREES = {
    "smpljoints": [
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
    ],
    "guoh3djoints": [  # no hands
        [0, 3, 6, 9, 12, 15],
        [9, 13, 16, 18, 20],
        [9, 14, 17, 19, 21],
        [0, 1, 4, 7, 10],
        [0, 2, 5, 8, 11],
    ],
}


@dataclass
class MatplotlibRender:
    jointstype: str = "smpljoints"
    fps: float = 20.0
    colors: List[str] = colors
    figsize: int = 4
    fontsize: int = 15
    canonicalize: bool = False

    def __call__(
        self,
        joints,
        highlights=None,
        title: str = "",
        output: str = "notebook",
        jointstype=None,
    ):
        jointstype = jointstype if jointstype is not None else self.jointstype
        render_animation(
            joints,
            title=title,
            highlights=highlights,
            output=output,
            jointstype=jointstype,
            fps=self.fps,
            colors=self.colors,
            figsize=(self.figsize, self.figsize),
            fontsize=self.fontsize,
            canonicalize=self.canonicalize,
        )


def init_axis(fig, title, radius=1.5):
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.view_init(elev=20.0, azim=-60)

    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])

    ax.set_aspect("auto")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()
    ax.grid(b=False)

    ax.set_title(title, loc="center", wrap=True)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Plot a plane XZ
    verts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz],
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    # Plot a bigger square plane XZ
    radius = max((maxx - minx), (maxy - miny))

    # center +- radius
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius

    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz],
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)
    return ax


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])


def render_animation(
    joints: np.ndarray,
    output: str = "notebook",
    highlights: Optional[np.ndarray] = None,
    jointstype: str = "smpljoints",
    title: str = "",
    fps: float = 20.0,
    colors: List[str] = colors,
    figsize: Tuple[int] = (4, 4),
    fontsize: int = 15,
    canonicalize: bool = False,
    agg=True,
):
    if agg:
        import matplotlib

        matplotlib.use("Agg")

    if highlights is not None:
        assert len(highlights) == len(joints)

    assert jointstype in KINEMATIC_TREES
    kinematic_tree = KINEMATIC_TREES[jointstype]

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe

    mean_fontsize = fontsize

    # heuristic to change fontsize
    fontsize = mean_fontsize - (len(title) - 30) / 20
    plt.rcParams.update({"font.size": fontsize})

    # Z is gravity here
    x, y, z = 0, 1, 2

    joints = joints.copy()

    if canonicalize:
        joints = canonicalize_rotation(joints, jointstype=jointstype)

    # Create a figure and initialize 3d plot
    fig = plt.figure(figsize=figsize)
    ax = init_axis(fig, title)

    # Create spline line
    trajectory = joints[:, 0, [x, y]]
    avg_segment_length = (
        np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    )
    draw_offset = int(25 / avg_segment_length)
    (spline_line,) = ax.plot(*trajectory.T, zorder=10, color="white")

    # Create a floor
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    # Put the character on the floor
    height_offset = np.min(joints[:, :, z])  # Min height
    joints = joints.copy()
    joints[:, :, z] -= height_offset

    # Initialization for redrawing
    lines = []
    initialized = False

    def update(frame):
        nonlocal initialized
        skeleton = joints[frame]

        root = skeleton[0]
        update_camera(ax, root)

        hcolors = colors
        if highlights is not None and highlights[frame]:
            hcolors = ("red", "red", "red", "red", "red")

        for index, (chain, color) in enumerate(
            zip(reversed(kinematic_tree), reversed(hcolors))
        ):
            if not initialized:
                lines.append(
                    ax.plot(
                        skeleton[chain, x],
                        skeleton[chain, y],
                        skeleton[chain, z],
                        linewidth=6.0,
                        color=color,
                        zorder=20,
                        path_effects=[pe.SimpleLineShadow(), pe.Normal()],
                    )
                )

            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])
                lines[index][0].set_color(color)

        left = max(frame - draw_offset, 0)
        right = min(frame + draw_offset, trajectory.shape[0])

        spline_line.set_xdata(trajectory[left:right, 0])
        spline_line.set_ydata(trajectory[left:right, 1])
        spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        initialized = True

    fig.tight_layout()
    frames = joints.shape[0]
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)

    if output == "notebook":
        from IPython.display import HTML

        HTML(anim.to_jshtml())
    else:
        # anim.save(output, writer='ffmpeg', fps=fps)
        anim.save(output, fps=fps)

    plt.close()
