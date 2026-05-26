"""I/O helpers: JSON saving, matplotlib scatter + bbox plotting."""
import json
import os
from dataclasses import is_dataclass, asdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve(path: str) -> str:
    """Resolve relative paths against the skill root, not the shell cwd.
    Absolute paths pass through unchanged."""
    return path if os.path.isabs(path) else os.path.join(SKILL_ROOT, path)

def save_json(obj, path: str):
    """Save a dict or dataclass instance to a JSON file. Creates parent dirs."""
    path = _resolve(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_scatter_plot(result, path: str, title_suffix: str = ""):
    """Save a 3D scatter plot of the reachable workspace point cloud.
    
    `result` must be a WorkspaceProbeResult (has .point_cloud, .bounds, .n_sampled).
    """
    path = _resolve(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pc = result.point_cloud

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.5, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    title = f"Reachable workspace (N={result.n_sampled})"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)

    _draw_bbox(ax, result.bounds)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_bbox(ax, bounds: dict):
    """Draw the 12 edges of an axis-aligned bounding box on a 3D axis."""
    x_lo, x_hi = bounds["x"]
    y_lo, y_hi = bounds["y"]
    z_lo, z_hi = bounds["z"]
    corners = [
        (x_lo, y_lo, z_lo), (x_hi, y_lo, z_lo),
        (x_hi, y_hi, z_lo), (x_lo, y_hi, z_lo),
        (x_lo, y_lo, z_hi), (x_hi, y_lo, z_hi),
        (x_hi, y_hi, z_hi), (x_lo, y_hi, z_hi),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    for i, j in edges:
        xs = [corners[i][0], corners[j][0]]
        ys = [corners[i][1], corners[j][1]]
        zs = [corners[i][2], corners[j][2]]
        ax.plot(xs, ys, zs, "r-", linewidth=1)