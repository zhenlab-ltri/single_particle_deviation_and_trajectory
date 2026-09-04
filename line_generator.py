import os
import cv2
import numpy as np
import pandas as pd
import h5py
import re
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from scipy.ndimage import gaussian_filter1d


def _boundary_along_rows(mask_2d, want_right):
    """Row-wise boundary scan excluding absolute image canvas borders."""
    h, w = mask_2d.shape
    has_pixel = np.any(mask_2d > 0, axis=1)
    full_row = np.all(mask_2d > 0, axis=1)
    valid_y = np.where(has_pixel & ~full_row)[0]
    if len(valid_y) == 0:
        return np.empty((0, 2)), 0.0, 0.0

    if want_right:
        flipped = mask_2d[:, ::-1]
        first_from_right = np.argmax(flipped > 0, axis=1)
        valid_x = (w - 1) - first_from_right[valid_y]
    else:
        valid_x = np.argmax(mask_2d > 0, axis=1)[valid_y]

    # Discard points touching canvas edges
    valid_mask = (valid_x > 0) & (valid_x < w - 1)
    valid_y = valid_y[valid_mask]
    valid_x = valid_x[valid_mask]

    if len(valid_y) == 0:
        return np.empty((0, 2)), 0.0, 0.0

    # First column = along-axis (y), second = across (x)
    pts = np.column_stack((valid_y.astype(np.float64), valid_x.astype(np.float64)))
    pts = pts[np.argsort(pts[:, 0])]
    return pts, pts[0, 0], pts[-1, 0]


def _boundary_along_columns(mask_2d, want_bottom):
    """Column-wise boundary scan excluding absolute image canvas borders."""
    h, w = mask_2d.shape
    has_pixel = np.any(mask_2d > 0, axis=0)
    full_column = np.all(mask_2d > 0, axis=0)
    valid_x = np.where(has_pixel & ~full_column)[0]
    if len(valid_x) == 0:
        return np.empty((0, 2)), 0.0, 0.0

    if want_bottom:
        flipped = mask_2d[::-1, :]
        first_from_bottom = np.argmax(flipped > 0, axis=0)
        valid_y = (h - 1) - first_from_bottom[valid_x]
    else:
        valid_y = np.argmax(mask_2d > 0, axis=0)[valid_x]

    # Discard points touching canvas edges
    valid_mask = (valid_y > 0) & (valid_y < h - 1)
    valid_x = valid_x[valid_mask]
    valid_y = valid_y[valid_mask]

    if len(valid_x) == 0:
        return np.empty((0, 2)), 0.0, 0.0

    # First column = along-axis (x), second = across (y)
    pts = np.column_stack((valid_x.astype(np.float64), valid_y.astype(np.float64)))
    pts = pts[np.argsort(pts[:, 0])]
    return pts, pts[0, 0], pts[-1, 0]


def _line_cleanliness(mask, along_rows):
    """Fraction of clean scanning lines."""
    lines = (mask > 0) if along_rows else (mask > 0).T
    padded = np.zeros((lines.shape[0], lines.shape[1] + 2), dtype=bool)
    padded[:, 1:-1] = lines
    d = np.diff(padded.astype(np.int8), axis=1)
    n_runs = np.sum(d == 1, axis=1)
    return float(np.mean(n_runs <= 1))


def extract_boundary_pair(upper_mask, lower_mask, prev_orientation=None, orientation_margin=1.1):
    """Auto-orienting boundary extraction pair (from Version 2)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_u = cv2.morphologyEx(upper_mask, cv2.MORPH_CLOSE, kernel)
    clean_u = cv2.morphologyEx(clean_u, cv2.MORPH_OPEN, kernel)
    clean_l = cv2.morphologyEx(lower_mask, cv2.MORPH_CLOSE, kernel)
    clean_l = cv2.morphologyEx(clean_l, cv2.MORPH_OPEN, kernel)

    uy_px, ux_px = np.where(clean_u > 0)
    ly_px, lx_px = np.where(clean_l > 0)

    if len(ux_px) == 0 or len(lx_px) == 0:
        return np.empty((0, 2)), 0.0, 0.0, np.empty((0, 2)), 0.0, 0.0, None

    row_score = (_line_cleanliness(clean_u, True) + _line_cleanliness(clean_l, True)) / 2.0
    col_score = (_line_cleanliness(clean_u, False) + _line_cleanliness(clean_l, False)) / 2.0

    vertical = row_score > col_score
    if prev_orientation == 'horizontal':
        vertical = row_score > col_score * orientation_margin
    elif prev_orientation == 'vertical':
        vertical = not (col_score > row_score * orientation_margin)

    if vertical:
        u_is_left = ux_px.mean() < lx_px.mean()
        upper_pts, u_min, u_max = _boundary_along_rows(clean_u, want_right=u_is_left)
        lower_pts, l_min, l_max = _boundary_along_rows(clean_l, want_right=not u_is_left)
        orientation = 'vertical'
    else:
        u_is_top = uy_px.mean() < ly_px.mean()
        upper_pts, u_min, u_max = _boundary_along_columns(clean_u, want_bottom=u_is_top)
        lower_pts, l_min, l_max = _boundary_along_columns(clean_l, want_bottom=not u_is_top)
        orientation = 'horizontal'

    if len(upper_pts) == 0 or len(lower_pts) == 0:
        return np.empty((0, 2)), 0.0, 0.0, np.empty((0, 2)), 0.0, 0.0, orientation

    return upper_pts, u_min, u_max, lower_pts, l_min, l_max, orientation


def centerline(upper_pts, lower_pts, num_nodes=2000, trig_harmonics=3, base_poly_degree=7):
    """
    Takes a sample of points across the frame and then:
    1) fits a base polynomial baseline from end to end (degree `base_poly_degree`;
       1 = straight line, higher degrees let the baseline itself bend), and
    2) computes a Fourier series (least squares trigonometric fit) on top of that
       baseline to add any further bending, using a parameter t to give points as (x(t), y(t)).

    Note: this only returns the curve spanning the observed upper/lower boundary
    points themselves. It does NOT extend the ends out to the frame's mask
    extents any more - that is a refinement-stage concern (see `extend_to_edges`),
    since extending against a raw, unsmoothed fit is what was causing the final
    line to fall short of / overshoot the true edge once refinement ran afterward.

    Args:
        base_poly_degree: degree of the polynomial fit used for the base line
            (np.polyfit deg). 1 gives the original straight-line baseline; higher
            values (e.g. 3) let the baseline itself curve before the harmonic
            (trig_harmonics) fit adds finer bending on top of it.
    """
    if len(upper_pts) < 15 or len(lower_pts) < 15:
        return None
        
    t_upper = np.linspace(0, 1, len(upper_pts))
    t_lower = np.linspace(0, 1, len(lower_pts))
    t_target = np.linspace(0, 1, num_nodes)
    
    try:
        ux = np.interp(t_target, t_upper, upper_pts[:, 0])
        uy = np.interp(t_target, t_upper, upper_pts[:, 1])
        lx = np.interp(t_target, t_lower, lower_pts[:, 0])
        ly = np.interp(t_target, t_lower, lower_pts[:, 1])
        
        cx = (ux + lx) / 2.0
        cy = (uy + ly) / 2.0
        
        base_poly_degree = max(1, min(base_poly_degree, num_nodes - 1))
        poly_coeff_x = np.polyfit(t_target, cx, deg=base_poly_degree)
        poly_coeff_y = np.polyfit(t_target, cy, deg=base_poly_degree)
        
        base_x = np.polyval(poly_coeff_x, t_target)
        base_y = np.polyval(poly_coeff_y, t_target)
        
        res_x = cx - base_x
        res_y = cy - base_y
        
        A = [np.ones_like(t_target)]
        for h in range(1, trig_harmonics + 1):
            A.append(np.sin(2 * np.pi * h * t_target))
            A.append(np.cos(2 * np.pi * h * t_target))
        A = np.column_stack(A)
        
        coeff_rx, _, _, _ = np.linalg.lstsq(A, res_x, rcond=None)
        coeff_ry, _, _, _ = np.linalg.lstsq(A, res_y, rcond=None)
        
        trig_res_x = A @ coeff_rx
        trig_res_y = A @ coeff_ry
        
        final_x = base_x + trig_res_x
        final_y = base_y + trig_res_y

        # Resample to uniform arc-length spacing (t_target above is uniform in
        # parameter space, not in physical distance) so downstream node spacing
        # is consistent. No edge-extension here anymore - see extend_to_edges.
        dx = np.diff(final_x)
        dy = np.diff(final_y)
        step_distances = np.sqrt(dx**2 + dy**2)

        cumulative_length = np.zeros(num_nodes)
        cumulative_length[1:] = np.cumsum(step_distances)

        total_length = cumulative_length[-1]
        if total_length <= 0:
            return np.column_stack((final_x, final_y))

        target_spacing = np.linspace(0, total_length, num_nodes)
        resampled_x = np.interp(target_spacing, cumulative_length, final_x)
        resampled_y = np.interp(target_spacing, cumulative_length, final_y)

        return np.column_stack((resampled_x, resampled_y))
        
    except Exception as e:
        print(f"Error calculating centerline: {e}")
        return None


def extend_to_edges(pts, target_min_x, target_max_x, num_ext_nodes=10):
    """
    Linearly extends a centerline's two ends out to the given target x bounds
    (typically the mask's observed x-extent for that frame), using the local
    tangent direction at each end, then re-resamples to uniform arc-length
    spacing with the same node count as the input.

    This is meant to run AFTER spatial refinement, not as part of the initial
    curve fit: extrapolating from the final, smoothed tangent (rather than the
    raw, unsmoothed fit) is what keeps the extension from being pulled back
    off the edge by a later damping/projection step, and from picking up a
    noisy tangent from the raw fit's last few points.

    Args:
        pts: (N, 2) array of (x, y) points, ordered by node_id.
        target_min_x: x value the proximal end should reach.
        target_max_x: x value the distal end should reach.
        num_ext_nodes: number of nodes used to build each extension segment
            before the final re-resampling to N nodes.
    Returns:
        (N, 2) array. Unchanged copy if there are fewer than 2 points, or if
        the endpoint tangents are vertical (zero x-extent) at that end.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n_nodes = len(pts)
    if n_nodes < 2:
        return pts.copy()

    tangent_span = min(20, n_nodes - 1)

    blocks = []

    vec_prox_x = pts[0, 0] - pts[tangent_span, 0]
    vec_prox_y = pts[0, 1] - pts[tangent_span, 1]
    if pts[0, 0] > target_min_x and vec_prox_x != 0:
        ext_x = np.linspace(target_min_x, pts[0, 0], num_ext_nodes, endpoint=False)
        slope_prox = vec_prox_y / vec_prox_x
        ext_y = pts[0, 1] + (ext_x - pts[0, 0]) * slope_prox
        blocks.append(np.column_stack((ext_x, ext_y)))

    blocks.append(pts)

    vec_dist_x = pts[-1, 0] - pts[-1 - tangent_span, 0]
    vec_dist_y = pts[-1, 1] - pts[-1 - tangent_span, 1]
    if pts[-1, 0] < target_max_x and vec_dist_x != 0:
        ext_x = np.linspace(pts[-1, 0], target_max_x, num_ext_nodes + 1)[1:]
        slope_dist = vec_dist_y / vec_dist_x
        ext_y = pts[-1, 1] + (ext_x - pts[-1, 0]) * slope_dist
        blocks.append(np.column_stack((ext_x, ext_y)))

    if len(blocks) == 1:
        return pts.copy()

    full_skeleton = np.vstack(blocks)

    dx = np.diff(full_skeleton[:, 0])
    dy = np.diff(full_skeleton[:, 1])
    step_distances = np.sqrt(dx**2 + dy**2)

    cumulative_length = np.zeros(len(full_skeleton))
    cumulative_length[1:] = np.cumsum(step_distances)

    total_length = cumulative_length[-1]
    if total_length <= 0:
        return pts.copy()

    target_spacing = np.linspace(0, total_length, n_nodes)
    resampled_x = np.interp(target_spacing, cumulative_length, full_skeleton[:, 0])
    resampled_y = np.interp(target_spacing, cumulative_length, full_skeleton[:, 1])

    return np.column_stack((resampled_x, resampled_y))


def extract_frame_number(filename):
    """
    Extracts frame number from filename.
    Args:
        filename: name of the file containing the masked frame
    Returns:
        The frame number.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None


def refine_frame_spatial(pts, scale_limit_px=3.0, spatial_sigma=1.2, curve_degree=8):
    """
    Applies the single-frame spatial refinement to one frame's centerline points.

    Rather than damping every point back toward a single rigid straight line
    (the major SVD eigenvector), this fits a low-order global polynomial curve
    (default degree 4, i.e. one global bend/peak shape - like x^4 about 0 -
    not a per-point wiggle) to the perpendicular offset as a function of
    position along the main axis, and damps deviations from THAT curve. This
    lets a real, global bow in the tissue survive refinement instead of being
    flattened into a straight line, while still cleaning up local noise/outliers
    that stick out past scale_limit_px from the smooth global shape.

    Steps:
    1) Get an axis/normal frame from SVD (used only to parameterize position
       along the frame, not as the reference shape itself).
    2) Fit an order-`curve_degree` polynomial of normal-offset vs. axis-position.
    3) tanh-damp each point's deviation from that fitted curve.
    4) Smooth along the node ordering with a gaussian filter.

    Args:
        pts: (N, 2) array of (x, y) centerline points, ordered by node_id.
        scale_limit_px: threshold pixel value (anything above this from the
            fitted curve will get damped)
        spatial_sigma: sigma value for the spatial gaussian filtering
        curve_degree: degree of the global polynomial reference curve. 1
            reproduces the old straight-line behavior; higher values (e.g. 4)
            let the reference curve itself bend/peak globally. Automatically
            capped so it can't exceed what the point count can support.
    Returns:
        (N, 2) array of refined points. If fewer than 5 points are given, returns
        an unchanged copy (matches the original per-frame behavior).
    """
    pts = np.asarray(pts, dtype=np.float64)

    if len(pts) < 5:
        return pts.copy()

    centroid = np.mean(pts, axis=0)
    centered_pts = pts - centroid
    _, _, Vh = np.linalg.svd(centered_pts, full_matrices=False)
    axis_dir = Vh[0]
    axis_dir /= np.linalg.norm(axis_dir)
    normal_dir = np.array([-axis_dir[1], axis_dir[0]])

    s = centered_pts @ axis_dir      # position along the main axis
    n = centered_pts @ normal_dir    # perpendicular offset from that axis

    # Normalize s before fitting so the polynomial is well-conditioned and
    # curve_degree behaves consistently regardless of the frame's physical length.
    s_scale = np.max(np.abs(s))
    s_norm = s / s_scale if s_scale > 1e-9 else s

    degree = max(1, min(curve_degree, len(pts) - 1))
    coeffs = np.polyfit(s_norm, n, deg=degree)
    n_fit = np.polyval(coeffs, s_norm)

    residual = n - n_fit
    distance = np.abs(residual)
    damped_distance = scale_limit_px * np.tanh(distance / scale_limit_px)
    n_adjusted = n_fit + np.sign(residual) * damped_distance

    adjusted_pts = centroid + s[:, None] * axis_dir[None, :] + n_adjusted[:, None] * normal_dir[None, :]

    smoothed_x = gaussian_filter1d(adjusted_pts[:, 0], sigma=spatial_sigma, mode='nearest')
    smoothed_y = gaussian_filter1d(adjusted_pts[:, 1], sigma=spatial_sigma, mode='nearest')

    return np.column_stack((smoothed_x, smoothed_y))


def refine_temporal(spatial_skeletons, temporal_sigma=0.5, batch_size=None):
    """
    Filters individual nodes by tracking them across multiple frames, smoothing
    each node's trajectory over time to drop major changes caused by improper masks.

    Args:
        spatial_skeletons: dict of {frame_id: (N, 2) array}, already spatially refined,
            all sharing the same number of nodes N and node ordering.
        temporal_sigma: sigma value for the temporal gaussian filtering
        batch_size: if given, the sorted frames are split into consecutive batches of
            at most this many frames, and each batch is smoothed independently (each
            batch's own edges use mode='nearest') instead of filtering across the whole
            frame range in one pass. This keeps a node's trajectory from being pulled by
            frames far away in time (which can look like a slow global shift), at the
            cost of a small discontinuity at each batch boundary. None (default) keeps
            the original whole-sequence behavior.
    Returns:
        dict of {frame_id: (N, 2) array}. Within a batch of 3 or fewer frames, or where
        frames in a batch don't share a common node count, that batch's frames are
        returned unchanged (matches the original per-node "only smooth if more than 3
        frames" behavior, applied per-batch).
    """
    frame_ids = sorted(spatial_skeletons.keys())

    if batch_size is None or batch_size <= 0 or batch_size >= len(frame_ids):
        batches = [frame_ids]
    else:
        batches = [frame_ids[i:i + batch_size] for i in range(0, len(frame_ids), batch_size)]

    result = {}
    for batch in batches:
        if len(batch) <= 3:
            for fid in batch:
                result[fid] = np.array(spatial_skeletons[fid], dtype=np.float64, copy=True)
            continue

        node_counts = {spatial_skeletons[fid].shape[0] for fid in batch}
        if len(node_counts) != 1:
            for fid in batch:
                result[fid] = np.array(spatial_skeletons[fid], dtype=np.float64, copy=True)
            continue

        stack_x = np.stack([spatial_skeletons[fid][:, 0] for fid in batch], axis=0)
        stack_y = np.stack([spatial_skeletons[fid][:, 1] for fid in batch], axis=0)

        smoothed_x = gaussian_filter1d(stack_x, sigma=temporal_sigma, axis=0, mode='nearest')
        smoothed_y = gaussian_filter1d(stack_y, sigma=temporal_sigma, axis=0, mode='nearest')

        for i, fid in enumerate(batch):
            result[fid] = np.column_stack((smoothed_x[i], smoothed_y[i]))

    return result


def refine_skeletons(skeleton_dict, target_bounds=None, orientations=None,
                     scale_limit_px=3.0, spatial_sigma=1.2,
                     curve_degree=4, temporal_sigma=0.5, temporal_batch_size=None):
    """
    Runs the full refinement (spatial per-frame -> edge extension -> temporal
    across frames) on an in-memory set of centerlines, without touching disk.
    This is the same refinement `refine_csv` applies, factored out so any
    caller (e.g. a live preview) can reproduce the exact same result the final
    CSV/video will have.

    Edge extension is deliberately run AFTER spatial refinement (not as part
    of the initial curve fit): extending from the final, smoothed tangent -
    rather than the raw fit - is what keeps the line landing edge-to-edge
    instead of being pulled back short by the damping step.

    For vertical orientation frames the primary axis is y. Points are temporarily
    swapped to (y, x) so that extend_to_edges (which works on the first coordinate)
    can be reused unchanged, then swapped back.

    Args:
        skeleton_dict: dict of {frame_id: (N, 2) array} of raw centerline points.
        target_bounds: optional dict of {frame_id: (target_min, target_max)}.
            When given, each frame's spatially-refined line is extended out to
            these bounds (along the primary axis) before temporal smoothing.
            Frames missing from this dict are left unextended. None (default)
            skips extension entirely.
        orientations: optional dict of {frame_id: 'horizontal'|'vertical'}.
            Required for correct extension when some frames are vertical.
        scale_limit_px: threshold pixel value for the spatial step.
        spatial_sigma: sigma value for the spatial gaussian filtering.
        curve_degree: degree of the global reference curve used in the spatial
            step (see refine_frame_spatial).
        temporal_sigma: sigma value for the temporal gaussian filtering.
        temporal_batch_size: if given, smooths the temporal step in consecutive batches
            of this many frames instead of across the whole sequence at once (see
            refine_temporal). None (default) smooths across all frames together.
    Returns:
        dict of {frame_id: (N, 2) array} of refined points.
    """
    spatial_skeletons = {
        fid: refine_frame_spatial(pts, scale_limit_px, spatial_sigma, curve_degree)
        for fid, pts in skeleton_dict.items()
    }

    if target_bounds is not None:
        extended = {}
        for fid, pts in spatial_skeletons.items():
            if fid not in target_bounds:
                extended[fid] = pts
                continue
            orient = orientations.get(fid, 'horizontal') if orientations is not None else 'horizontal'
            if orient == 'vertical':
                # Work in (y, x) so primary axis is first coordinate
                swapped = pts[:, [1, 0]]
                ext = extend_to_edges(swapped, *target_bounds[fid])
                extended[fid] = ext[:, [1, 0]]
            else:
                extended[fid] = extend_to_edges(pts, *target_bounds[fid])
        spatial_skeletons = extended

    return refine_temporal(spatial_skeletons, temporal_sigma, temporal_batch_size)


def refine_csv(file_path, scale_limit_px=3.0, spatial_sigma=1.2, curve_degree=4,
               temporal_sigma=0.5, temporal_batch_size=None):
    """
    Modifies the coordinates in the csv file as follows:
    1. Damps points that are > scale_limit_px from a global reference curve
       (fit per-frame; see refine_frame_spatial) in a single frame.
    2. Extends each frame's line out to that frame's recorded mask extent
       (frame_min_x/frame_max_x columns), using the refined line's own end
       tangents (see extend_to_edges). Frames without those columns are left
       as-is (older CSVs without them will just skip extension).
       For vertical frames the stored bounds are the primary-axis (y) extents;
       points are temporarily swapped so extension still works correctly.
    3. Filters individual nodes by tracking them across multiple frames to
       drop major changes caused by improper masks.
    Args:
        file_path: path to the csv file
        scale_limit_px: threshold pixel value (anything above this from the
            reference curve will get damped)
        spatial_sigma: sigma value for gaussian filtering used in step 1
        curve_degree: degree of the global reference curve used in step 1
        temporal_sigma: sigma value for gaussian filtering used in step 3
        temporal_batch_size: if given, applies step 3 in consecutive batches of this many
            frames instead of across the whole sequence at once (see refine_temporal).
    No return value.
    """
    df = pd.read_csv(file_path)

    frame_groups = {}
    order = []
    for frame_id, group in df.groupby('frame_id'):
        sorted_nodes = group.sort_values('node_id').copy()
        frame_groups[frame_id] = sorted_nodes
        order.append(frame_id)

    skeleton_dict = {
        fid: frame_groups[fid][['x_pixel', 'y_pixel']].to_numpy(dtype=np.float64)
        for fid in order
    }

    target_bounds = None
    orientations = None
    if 'frame_min_x' in df.columns and 'frame_max_x' in df.columns:
        target_bounds = {
            fid: (frame_groups[fid]['frame_min_x'].iloc[0], frame_groups[fid]['frame_max_x'].iloc[0])
            for fid in order
        }
        if 'orientation' in df.columns:
            orientations = {
                fid: frame_groups[fid]['orientation'].iloc[0]
                for fid in order
            }

    refined = refine_skeletons(skeleton_dict, target_bounds, orientations,
                               scale_limit_px, spatial_sigma,
                               curve_degree, temporal_sigma, temporal_batch_size)

    out_rows = []
    for fid in order:
        sorted_nodes = frame_groups[fid]
        pts = refined[fid]
        sorted_nodes['x_pixel'] = pts[:, 0]
        sorted_nodes['y_pixel'] = pts[:, 1]
        out_rows.append(sorted_nodes)

    final_df = pd.concat(out_rows).sort_values(['frame_id', 'node_id'])
    final_df.to_csv(file_path, index=False)
    

def create_csv(folders, boundary, input_path, output_path):
    """
    Writes a csv file that is (N * num_nodes) x 6, each row containing 
    frame number, node id, x coordinate, y coordinate, the global node spacing,
    the primary-axis mask extents, and the detected orientation.
    Args:
        folders: paths to folders with the upper masks and lower masks
        boundary: ['upper', 'lower'] for computing the specific edge of each mask
        input_path: path to the original h5 video
        output_path: path to the folder to which the csv gets saved
    No return value.
    """
    with h5py.File(input_path, 'r') as h5_in:
        detected_video_key = list(h5_in.keys())[0]
        total_video_frames, img_height, width = h5_in[detected_video_key].shape[:3]

    upper_idx = boundary.index('upper') if 'upper' in boundary else None
    lower_idx = boundary.index('lower') if 'lower' in boundary else None

    upper_files = sorted([f for f in os.listdir(folders[upper_idx]) if f.endswith(('.png', '.jpg'))])
    lower_files = sorted([f for f in os.listdir(folders[lower_idx]) if f.endswith(('.png', '.jpg'))])

    with open(output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'node_id', 'x_pixel', 'y_pixel', 'node_spacing',
                              'frame_min_x', 'frame_max_x', 'orientation'])
    
        prev_orientation = None
        for u_file, l_file in zip(upper_files, lower_files):
            frame_idx = extract_frame_number(u_file)
            if frame_idx is None or frame_idx >= total_video_frames:
                continue
                
            img_u = cv2.imread(os.path.join(folders[upper_idx], u_file), cv2.IMREAD_GRAYSCALE)
            img_l = cv2.imread(os.path.join(folders[lower_idx], l_file), cv2.IMREAD_GRAYSCALE)
            
            if img_u is None or img_l is None:
                continue
                
            _, mask_u = cv2.threshold(img_u, 1, 255, cv2.THRESH_BINARY)
            _, mask_l = cv2.threshold(img_l, 1, 255, cv2.THRESH_BINARY)
            
            upper_pts, u_min, u_max, lower_pts, l_min, l_max, orientation = extract_boundary_pair(
                mask_u, mask_l, prev_orientation=prev_orientation
            )
            if orientation is not None:
                prev_orientation = orientation
            
            # Points arrive already ordered by primary axis:
            #   horizontal → (x, y)
            #   vertical   → (y, x)
            skeleton_nodes = centerline(upper_pts, lower_pts,
                                        num_nodes=2000, trig_harmonics=3, base_poly_degree=3)
            
            if skeleton_nodes is not None:
                # Bring vertical results back to image coordinates (x, y)
                if orientation == 'vertical':
                    skeleton_nodes = skeleton_nodes[:, [1, 0]]

                dx = np.diff(skeleton_nodes[:, 0])
                dy = np.diff(skeleton_nodes[:, 1])
                step_distances = np.sqrt(dx**2 + dy**2)
                node_spacing = np.mean(step_distances)

                # Primary-axis extents (x for horizontal, y for vertical).
                # Stored under the same column names so the post-refinement
                # extend_to_edges path stays unchanged; orientation tells the
                # refinement stage how to interpret them.
                frame_min_primary = min(u_min, l_min)
                frame_max_primary = max(u_max, l_max)
                
                for node_idx, (x_val, y_val) in enumerate(skeleton_nodes):
                    safe_x = np.clip(x_val, 0, width - 1)
                    safe_y = np.clip(y_val, 0, img_height - 1)
                    csv_writer.writerow([frame_idx, node_idx, f"{safe_x:.4f}", f"{safe_y:.4f}",
                                          f"{node_spacing:.6f}",
                                          f"{frame_min_primary:.4f}", f"{frame_max_primary:.4f}",
                                          orientation if orientation is not None else 'horizontal'])


def create_overlay_video(input_source, csv_coordinates_path, output_video_path, fps=30):
    df_skeleton = pd.read_csv(csv_coordinates_path)
    skeleton = {frame_id: group[['x_pixel', 'y_pixel']].to_numpy() for frame_id, group in df_skeleton.groupby('frame_id')}
    
    with h5py.File(input_source, 'r') as h5_in:
        video_key = list(h5_in.keys())[0]
        dataset = h5_in[video_key]
        total_frames, img_height, width = dataset.shape[:3]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, img_height))
        
        valid_frames = sorted(skeleton.keys())
        
        for frame_idx in valid_frames:
            if frame_idx >= total_frames or frame_idx < 0:
                continue
                
            raw_frame = np.array(dataset[frame_idx])
            color_frame = cv2.cvtColor(raw_frame, cv2.COLOR_GRAY2BGR)
            
            nodes = skeleton[frame_idx]
            for i in range(len(nodes) - 1):
                p1 = (int(round(nodes[i][0])), int(round(nodes[i][1])))
                p2 = (int(round(nodes[i+1][0])), int(round(nodes[i+1][1])))
                if (0 <= p1[0] < width and 0 <= p1[1] < img_height and 
                    0 <= p2[0] < width and 0 <= p2[1] < img_height):
                    cv2.line(color_frame, p1, p2, (0, 0, 255), 1)
                    
            video_writer.write(color_frame)
            
        video_writer.release()


class SkeletonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Centerline Generator")
        self.root.geometry("700x250")
        self.upper_folder = tk.StringVar()
        self.lower_folder = tk.StringVar()
        self.hdf5_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.create_widgets()

    def create_widgets(self):
        paths_frame = ttk.LabelFrame(self.root, text="Configurations", padding=10)
        paths_frame.pack(fill="x", padx=15, pady=8)
        
        self.add_path_row(paths_frame, "Upper Mask Folder:", self.upper_folder, is_dir=True)
        self.add_path_row(paths_frame, "Lower Mask Folder:", self.lower_folder, is_dir=True)
        self.add_path_row(paths_frame, "Original Video (HDF5 / .h5):", self.hdf5_path, is_dir=False, file_types=[("HDF5 files", "*.h5 *.hdf5")])
        self.add_path_row(paths_frame, "Output Directory:", self.output_dir, is_dir=True)

        run_frame = ttk.Frame(self.root, padding=5)
        run_frame.pack(fill="x", padx=15, pady=5)
        self.btn_run = ttk.Button(run_frame, text="Generate Centerline", command=self.start_processing_thread)
        self.btn_run.pack(fill="x", ipady=6)

    def add_path_row(self, parent, label_text, var, is_dir, file_types=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label_text, width=28, anchor="w").pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=5)
        btn_text = "Browse Folder" if is_dir else "Browse File"
        ttk.Button(row, text=btn_text, command=lambda: self.browse_path(var, is_dir, file_types)).pack(side="right")

    def browse_path(self, var, is_dir, file_types):
        path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename(filetypes=file_types)
        if path: var.set(path)

    def start_processing_thread(self):
        if not self.upper_folder.get() or not self.lower_folder.get():
            messagebox.showerror("Error", "Both upper and lower background mask folders are required.")
            return
        if not self.hdf5_path.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Verify execution paths.")
            return
        threading.Thread(target=self.run_pipeline, daemon=True).start()

    def run_pipeline(self):
        self.btn_run.configure(state="disabled")
        
        folders = [self.upper_folder.get(), self.lower_folder.get()]
        boundary = ['upper', 'lower']

        output_csv_path = os.path.join(self.output_dir.get(), 'pharynx_axis_coordinates.csv')
        output_video_path = os.path.join(self.output_dir.get(), 'pharynx_skeleton_overlay.mp4')
        
        try:
            create_csv(folders, boundary, self.hdf5_path.get(), output_csv_path)
            refine_csv(output_csv_path, scale_limit_px=3.0, spatial_sigma=1.2, temporal_sigma=0.5,
                       temporal_batch_size=5)
            create_overlay_video(self.hdf5_path.get(), output_csv_path, output_video_path)
            messagebox.showinfo("Success", "Centerline saved to output directory.")
        except Exception as e:
            messagebox.showerror("Execution Fault", f"Pipeline failed:\n{str(e)}")
        finally:
            self.btn_run.configure(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = SkeletonApp(root)
    root.mainloop()
