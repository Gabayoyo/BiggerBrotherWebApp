import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation
import numpy as np

# MediaPipe pose connections (landmark index pairs)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

def animate_skeleton(frame_data_list, save_path=None, fps=30):
    """
    Create a 3D animation of pose landmarks with missing‑data‑friendly smoothing.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Precompute all keypoints (frames x landmarks x 3)
    all_points = np.array([
        [lm.to_array() for lm in fd.world_landmarks]
        for fd in frame_data_list
    ])   # shape (n_frames, 33, 3)

    # Precompute visibility (frames x landmarks)
    all_visibility = np.array([
        [lm.visibility for lm in fd.world_landmarks]
        for fd in frame_data_list
    ])   # shape (n_frames, 33)

    n_frames, n_landmarks, n_dims = all_points.shape

    # Mask out low-visibility points as NaN
    low_vis_mask = all_visibility < 0.5
    all_points[low_vis_mask] = np.nan

    # Interpolate NaNs along the frame axis for each landmark/dimension
    for lm in range(n_landmarks):
        for d in range(n_dims):
            series = all_points[:, lm, d]
            nan_mask = np.isnan(series)

            if nan_mask.all():
                # No valid data at all for this landmark/dim — zero it out
                series[:] = 0.0
                all_points[:, lm, d] = series
                continue
            if nan_mask.any():
                valid = ~nan_mask
                series[nan_mask] = np.interp(
                    np.flatnonzero(nan_mask),
                    np.flatnonzero(valid),
                    series[valid]
                )
                all_points[:, lm, d] = series

    # Savgol filter requires window_length <= n_frames and odd
    all_points = savgol_filter(all_points, window_length=5, polyorder=1, axis=0)

    # Determine consistent axis limits across all frames
    all_coords = all_points.reshape(-1, 3)
    margin = 0.2
    x_min, x_max = all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin
    y_min, y_max = all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin
    z_min, z_max = all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin

    scatter = ax.scatter([], [], [], c='red', s=20)
    lines = [ax.plot([], [], [], 'b-')[0] for _ in POSE_CONNECTIONS]
    title = ax.set_title("")

    def init():
        ax.set_xlim(-x_max, -x_min)
        ax.set_ylim(z_min, z_max)   # swap y/z for more natural "depth" view
        ax.set_zlim(-y_max, -y_min) # invert y since image coords go down
        ax.view_init(elev=0, azim=-90)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scatter, *lines, title]

    def update(frame_idx):
        points = all_points[frame_idx]
        xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]

        # Remap for plotting: (x, z, -y)
        scatter._offsets3d = (xs, zs, -ys)

        for line, (i, j) in zip(lines, POSE_CONNECTIONS):
            line.set_data([xs[i], xs[j]], [zs[i], zs[j]])
            line.set_3d_properties([-ys[i], -ys[j]])

        fd = frame_data_list[frame_idx]
        title.set_text(f"Frame {fd.frame_number} | t={fd.timestamp_s:.2f}s")
        return [scatter, *lines, title]

    anim = FuncAnimation(
        fig, update, frames=len(frame_data_list),
        init_func=init, interval=1000 / fps, blit=False
    )

    if save_path:
        anim.save(save_path, fps=fps)
    else:
        plt.show()

    return anim


# Usage:
# anim = animate_skeleton(frame_data_list, save_path="skeleton.mp4", fps=30)