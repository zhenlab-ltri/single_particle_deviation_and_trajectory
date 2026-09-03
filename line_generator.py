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


def extract_boundary_points(binary_mask, boundary_type):
    """
    Extracts the upper boundary of frames of lower mask and the lower boundary of frames of upper mask.
    Columns that are fully colored top-to-bottom (no real edge, just a solid
    band) are ignored and excluded from the returned points.
    Args:
    - binary_mask: The particular frame with mask
    - boundary_type: Determines whether to extract the upper edge or the lower edge
    Returns the x and y coordinates of the boundary in a 2 dimensional array (N x 2), the maximum and minimum x value where the mask exists.

    NOTE: this slices strictly by image column (fixed x, scan down y), which
    only gives a well-defined single boundary point per column when the tube
    runs roughly left-to-right. For tubes at an arbitrary angle (including
    close to vertical), use extract_boundary_pair instead, which fits a
    principal axis via PCA and slices perpendicular to *that* instead of a
    fixed image axis. Kept here for reference / backward compatibility.
    """
    h, w = binary_mask.shape
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    has_pixel   = np.any(cleaned_mask > 0, axis=0)
    full_column = np.all(cleaned_mask > 0, axis=0)
    has_pixel   = has_pixel & ~full_column
    valid_x = np.where(has_pixel)[0]
    
    if len(valid_x) == 0:
        return np.empty((0, 2)), 0, 0

    if boundary_type == 'upper':
        flipped_mask = cleaned_mask[::-1, :]
        first_y_from_bottom = np.argmax(flipped_mask > 0, axis=0)
        valid_y = (h - 1) - first_y_from_bottom[valid_x]
    else:
        valid_y = np.argmax(cleaned_mask > 0, axis=0)[valid_x]
        
    points = np.column_stack((valid_x, valid_y.astype(np.float64)))
    
    sort_idx = np.argsort(points[:, 0])
    points = points[sort_idx]
    
    raw_min_x = points[0, 0]
    raw_max_x = points[-1, 0]
        
    return points, raw_min_x, raw_max_x


from collections import namedtuple

# Bundles an origin together with an axis direction. s = (p - origin) .
# direction. For the axis-snapped extraction below, origin is always (0, 0)
# and direction is always exactly (1, 0) or (0, 1), so s reduces to plain x
# or plain y -- kept as a named pair (rather than returning a bare x/y flag)
# so centerline's projection code doesn't need a special case.
AxisInfo = namedtuple('AxisInfo', ['origin', 'direction'])


def _boundary_along_columns(mask_2d, want_bottom):
    """
    Exact per-column boundary extraction -- identical to the logic in
    extract_boundary_points, but returns (s, x, y) triples (with s == x)
    instead of (x, y) pairs, and takes a plain want_bottom flag instead of a
    named boundary_type, so the caller can resolve "which mask is on top"
    generically via mask position rather than assuming the upper_mask
    parameter is always geometrically above the lower_mask parameter.
    """
    h, w = mask_2d.shape
    has_pixel = np.any(mask_2d > 0, axis=0)
    full_column = np.all(mask_2d > 0, axis=0)
    valid_x = np.where(has_pixel & ~full_column)[0]
    if len(valid_x) == 0:
        return np.empty((0, 3))

    if want_bottom:
        flipped = mask_2d[::-1, :]
        first_from_bottom = np.argmax(flipped > 0, axis=0)
        valid_y = (h - 1) - first_from_bottom[valid_x]
    else:
        valid_y = np.argmax(mask_2d > 0, axis=0)[valid_x]

    s = valid_x.astype(np.float64)
    pts = np.column_stack((s, s, valid_y.astype(np.float64)))
    return pts[np.argsort(pts[:, 0])]


def _boundary_along_rows(mask_2d, want_right):
    """
    Row-wise mirror of _boundary_along_columns, for tubes that run mostly
    vertically: scans across x for each row instead of down y for each
    column. Returns (s, x, y) triples with s == y.
    """
    h, w = mask_2d.shape
    has_pixel = np.any(mask_2d > 0, axis=1)
    full_row = np.all(mask_2d > 0, axis=1)
    valid_y = np.where(has_pixel & ~full_row)[0]
    if len(valid_y) == 0:
        return np.empty((0, 3))

    if want_right:
        flipped = mask_2d[:, ::-1]
        first_from_right = np.argmax(flipped > 0, axis=1)
        valid_x = (w - 1) - first_from_right[valid_y]
    else:
        valid_x = np.argmax(mask_2d > 0, axis=1)[valid_y]

    s = valid_y.astype(np.float64)
    pts = np.column_stack((s, valid_x.astype(np.float64), s))
    return pts[np.argsort(pts[:, 0])]


def extract_boundary_pair(upper_mask, lower_mask, prev_orientation=None,
                           vertical_angle_deg=45.0, orientation_margin=1.2):
    """
    Auto-orienting replacement for calling extract_boundary_points on the
    upper and lower masks separately: estimates the tube's tilt from the
    combined upper+lower mask's bounding box, then slices by exact pixel
    column (the original left-to-right behavior) unless the tube is tilted
    more steeply than `vertical_angle_deg` from horizontal, in which case it
    switches to an exact pixel row scan instead -- so ordinary and
    moderately-tilted tubes get exactly the original per-column precision,
    and only genuinely close-to-vertical tubes get the row-wise treatment.

    For each column/row, the inner-facing edge is kept: whichever mask is
    "above" (or "left of") the other has its bottom (or right) edge taken,
    and vice versa for the other mask -- the same "facing" pairing
    extract_boundary_points used, just resolved from the masks' relative
    position instead of assumed from parameter naming, so it stays correct
    even if upper_mask/lower_mask are swapped.

    Args:
        upper_mask, lower_mask: binary masks, same shape.
        prev_orientation: optional 'horizontal' or 'vertical' from the
            previous frame. Right around the vertical_angle_deg boundary,
            orientation could otherwise flip frame-to-frame and introduce
            jitter; passing the previous frame's orientation adds
            hysteresis -- an already-chosen orientation is kept unless the
            new frame's tilt crosses the boundary by more than
            `orientation_margin`. Pass None for a standalone, single-frame
            call (e.g. a one-off preview) where cross-frame consistency
            doesn't apply.
        vertical_angle_deg: tilt from horizontal (in degrees, using the
            mask's bounding-box aspect ratio as a proxy for tilt) beyond
            which the tube is treated as vertical instead of horizontal.
            E.g. the default 75 means anything tilted 75 degrees or steeper
            switches to a row-wise scan; anything up to 75 degrees still
            uses the standard column-wise scan.
        orientation_margin: how decisively the aspect ratio must cross the
            vertical_angle_deg boundary before switching (e.g. 1.2 means
            20% past the boundary ratio) -- only used when prev_orientation
            is given.

    Returns:
        upper_pts, u_min_s, u_max_s, lower_pts, l_min_s, l_max_s, axis
        where upper_pts/lower_pts are (N, 3) arrays of (s, x, y) sorted by
        s (s == x for a horizontal frame, s == y for a vertical one), and
        (x, y) are real pixel coordinates. u_min_s/u_max_s/l_min_s/l_max_s
        are each mask's s-extent. axis is an AxisInfo(origin, direction)
        with direction (1, 0) or (0, 1) -- pass it into centerline's `axis`
        argument, and pass the orientation ('horizontal'/'vertical', which
        you can read off which component of axis.direction is nonzero) back
        in as this function's `prev_orientation` on the next frame. Returns
        empty points (and axis=None) if either mask has no usable pixels.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_u = cv2.morphologyEx(upper_mask, cv2.MORPH_CLOSE, kernel)
    clean_u = cv2.morphologyEx(clean_u, cv2.MORPH_OPEN, kernel)
    clean_l = cv2.morphologyEx(lower_mask, cv2.MORPH_CLOSE, kernel)
    clean_l = cv2.morphologyEx(clean_l, cv2.MORPH_OPEN, kernel)

    uy_px, ux_px = np.where(clean_u > 0)
    ly_px, lx_px = np.where(clean_l > 0)

    if len(ux_px) == 0 or len(lx_px) == 0:
        return np.empty((0, 3)), 0.0, 0.0, np.empty((0, 3)), 0.0, 0.0, None

    all_x = np.concatenate([ux_px, lx_px])
    all_y = np.concatenate([uy_px, ly_px])
    width_extent = float(all_x.max() - all_x.min())
    height_extent = float(all_y.max() - all_y.min())

    # For a straight segment of length L tilted `theta` from horizontal,
    # width_extent ~ L*cos(theta) and height_extent ~ L*sin(theta), so
    # height/width ~ tan(theta) -- comparing against tan(vertical_angle_deg)
    # (rather than comparing width vs. height directly, which is implicitly
    # a 45-degree threshold) makes the switch happen at the requested angle.
    angle_ratio = np.tan(np.radians(vertical_angle_deg))

    vertical = height_extent > width_extent * angle_ratio
    if prev_orientation == 'horizontal':
        vertical = height_extent > width_extent * angle_ratio * orientation_margin
    elif prev_orientation == 'vertical':
        vertical = height_extent > width_extent * angle_ratio / orientation_margin

    if vertical:
        # Whichever mask sits further left has its inner edge on the right
        # (facing the other mask), and vice versa.
        u_is_left = ux_px.mean() < lx_px.mean()
        upper_pts = _boundary_along_rows(clean_u, want_right=u_is_left)
        lower_pts = _boundary_along_rows(clean_l, want_right=not u_is_left)
        axis = AxisInfo(origin=np.array([0.0, 0.0]), direction=np.array([0.0, 1.0]))
    else:
        u_is_top = uy_px.mean() < ly_px.mean()
        upper_pts = _boundary_along_columns(clean_u, want_bottom=u_is_top)
        lower_pts = _boundary_along_columns(clean_l, want_bottom=not u_is_top)
        axis = AxisInfo(origin=np.array([0.0, 0.0]), direction=np.array([1.0, 0.0]))

    if len(upper_pts) == 0 or len(lower_pts) == 0:
        return np.empty((0, 3)), 0.0, 0.0, np.empty((0, 3)), 0.0, 0.0, axis

    return (upper_pts, upper_pts[0, 0], upper_pts[-1, 0],
            lower_pts, lower_pts[0, 0], lower_pts[-1, 0], axis)


import numpy as np

def _eval_fitted_axis(t, lin_coeff, harmonic_coeff, trig_harmonics):
    """Evaluate a fitted base-polynomial + trig-harmonic axis at arbitrary t.

    Because both the polynomial and the sinusoids are defined everywhere (not
    just on the [0, 1] domain they were fit over), this can be called with t
    values outside [0, 1] to continue the fitted curve's shape rather than
    just its local tangent -- used to build curved end extensions.
    """
    base = np.polyval(lin_coeff, t)
    A_terms = [np.ones_like(t)]
    for h in range(1, trig_harmonics + 1):
        A_terms.append(np.sin(2 * np.pi * h * t))
        A_terms.append(np.cos(2 * np.pi * h * t))
    A = np.column_stack(A_terms)
    return base + A @ harmonic_coeff


def centerline(upper_pts, u_min_s, u_max_s, lower_pts, l_min_s, l_max_s,
                num_nodes=2000, trig_harmonics=3, base_degree=1, axis=None):
    """
    Takes a sample of points across the frame and then:
    1) restricts to the axis-bins where *both* the upper and lower boundary
       have a point (bins where only one of the two exists are dropped,
       since a centerline can't be meaningfully averaged there),
    2) fits a baseline of degree `base_degree` from end to end over that
       common-bin region, and
    3) computes a Fourier series (least squares trigonometric fit) to add any bending
       using a parameter t to give points as (x(t), y(t)).
    4) extends both ends out to the full mask extent (the union of the upper
       and lower masks' extents along the principal axis, not just the
       common region) by continuing the *same* fitted base+harmonic curve
       past t=0 / t=1 (rather than a straight-line extrapolation), so the
       extension keeps whatever curvature the fit has.

    Args:
        upper_pts, lower_pts: (N, 3) arrays of (s, x, y) as returned by
            extract_boundary_pair -- s is the 1-D coordinate along the
            tube's principal axis (used for matching/extent, playing the
            role plain x played when the tube was assumed left-to-right),
            and (x, y) are the real pixel coordinates that get fitted.
            (N, 2) arrays of plain (x, y) are also accepted for backward
            compatibility, treating x as s -- equivalent to the old
            left-to-right-only behavior.
        u_min_s, u_max_s, l_min_s, l_max_s: each mask's extent along s, as
            returned by extract_boundary_pair.
        base_degree: degree of the polynomial baseline fit (1 = straight line,
            higher = allows the baseline itself to bend before the trig
            harmonics are added on top).
        axis: the AxisInfo(origin, direction) returned by
            extract_boundary_pair. Needed so the fitted (x, y) curve can be
            re-projected onto s (s = (p - origin) . direction) to know how
            far past each end to extend -- both the origin and the
            direction are required, since projecting without first
            subtracting the origin introduces an offset equal to the
            origin's own projection. If None, the fitted x-coordinate is
            used directly as a stand-in for s (correct only for the
            left-to-right case).
    """
    if len(upper_pts) < 15 or len(lower_pts) < 15:
        return None

    upper_pts = np.asarray(upper_pts, dtype=np.float64)
    lower_pts = np.asarray(lower_pts, dtype=np.float64)
    x_col, y_col = (1, 2) if upper_pts.shape[1] >= 3 else (0, 1)

    # Only keep axis-bins where both the upper and lower boundary have a
    # point. extract_boundary_pair returns one point per bin, so matching on
    # s directly (rather than by index/arc-length) correctly pairs up the
    # two boundaries bin-for-bin.
    upper_s = np.round(upper_pts[:, 0]).astype(np.int64)
    lower_s = np.round(lower_pts[:, 0]).astype(np.int64)
    common_s, upper_idx, lower_idx = np.intersect1d(upper_s, lower_s, return_indices=True)

    if len(common_s) < 15:
        return None

    order = np.argsort(common_s)
    matched_upper = upper_pts[upper_idx][order]
    matched_lower = lower_pts[lower_idx][order]

    t_common = np.linspace(0, 1, len(common_s))
    t_target = np.linspace(0, 1, num_nodes)

    try:
        ux = np.interp(t_target, t_common, matched_upper[:, x_col])
        uy = np.interp(t_target, t_common, matched_upper[:, y_col])
        lx = np.interp(t_target, t_common, matched_lower[:, x_col])
        ly = np.interp(t_target, t_common, matched_lower[:, y_col])

        cx = (ux + lx) / 2.0
        cy = (uy + ly) / 2.0

        lin_coeff_x = np.polyfit(t_target, cx, deg=base_degree)
        lin_coeff_y = np.polyfit(t_target, cy, deg=base_degree)

        base_x = np.polyval(lin_coeff_x, t_target)
        base_y = np.polyval(lin_coeff_y, t_target)

        res_x = cx - base_x
        res_y = cy - base_y

        A = [np.ones_like(t_target)]
        for h in range(1, trig_harmonics + 1):
            A.append(np.sin(2 * np.pi * h * t_target))
            A.append(np.cos(2 * np.pi * h * t_target))
        A = np.column_stack(A)

        coeff_rx, _, _, _ = np.linalg.lstsq(A, res_x, rcond=None)
        coeff_ry, _, _, _ = np.linalg.lstsq(A, res_y, rcond=None)

        final_x = base_x + A @ coeff_rx
        final_y = base_y + A @ coeff_ry

        # Re-project the fitted curve onto the principal axis to get its own
        # s-coordinate, so extension distance is measured along the tube's
        # actual direction rather than along image x (which would be wrong
        # for a vertical or diagonal tube). Must subtract the origin first --
        # projecting the raw (x, y) values would offset s by the origin's
        # own projection.
        if axis is not None:
            origin_x, origin_y = axis.origin
            dir_x, dir_y = axis.direction
            final_s = (final_x - origin_x) * dir_x + (final_y - origin_y) * dir_y
        else:
            final_s = final_x

        target_start_s = min(u_min_s, l_min_s)
        target_end_s = max(u_max_s, l_max_s)

        dt = t_target[1] - t_target[0]
        idx_span = min(20, num_nodes - 1)
        num_ext_nodes = 10

        # Proximal extension: continue the fitted curve backward past t=0.
        extended_proximal = np.empty((0, 2))
        ds_span_p = final_s[idx_span] - final_s[0]
        if final_s[0] > target_start_s and abs(ds_span_p) > 1e-9:
            dt_span_p = t_target[idx_span] - t_target[0]
            delta_t = (target_start_s - final_s[0]) * dt_span_p / ds_span_p  # negative: extends before t=0
            t_ext = np.linspace(delta_t, 0, num_ext_nodes, endpoint=False)
            ext_x = _eval_fitted_axis(t_ext, lin_coeff_x, coeff_rx, trig_harmonics)
            ext_y = _eval_fitted_axis(t_ext, lin_coeff_y, coeff_ry, trig_harmonics)
            extended_proximal = np.column_stack((ext_x, ext_y))

        # Distal extension: continue the fitted curve forward past t=1.
        extended_distal = np.empty((0, 2))
        ds_span_d = final_s[-1] - final_s[-1 - idx_span]
        if final_s[-1] < target_end_s and abs(ds_span_d) > 1e-9:
            dt_span_d = t_target[-1] - t_target[-1 - idx_span]
            delta_t = (target_end_s - final_s[-1]) * dt_span_d / ds_span_d
            t_ext = np.linspace(1, 1 + delta_t, num_ext_nodes + 1)[1:]
            ext_x = _eval_fitted_axis(t_ext, lin_coeff_x, coeff_rx, trig_harmonics)
            ext_y = _eval_fitted_axis(t_ext, lin_coeff_y, coeff_ry, trig_harmonics)
            extended_distal = np.column_stack((ext_x, ext_y))

        core_nodes = np.column_stack((final_x, final_y))

        blocks = []
        if len(extended_proximal) > 0:
            blocks.append(extended_proximal)
        blocks.append(core_nodes)
        if len(extended_distal) > 0:
            blocks.append(extended_distal)

        full_skeleton = np.vstack(blocks)

        dx = np.diff(full_skeleton[:, 0])
        dy = np.diff(full_skeleton[:, 1])
        step_distances = np.sqrt(dx**2 + dy**2)

        cumulative_length = np.zeros(len(full_skeleton))
        cumulative_length[1:] = np.cumsum(step_distances)

        total_length = cumulative_length[-1]
        target_spacing = np.linspace(0, total_length, num_nodes)

        resampled_x = np.interp(target_spacing, cumulative_length, full_skeleton[:, 0])
        resampled_y = np.interp(target_spacing, cumulative_length, full_skeleton[:, 1])

        return np.column_stack((resampled_x, resampled_y))

    except Exception as e:
        print(f"Error calculating centerline: {e}")
        return None

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


def refine_frame_points(points, scale_limit_px=3.0, spatial_sigma=1.2, backbone_sigma=15.0):
    """
    Applies per-frame spatial refinement while preserving the centerline's natural bend.

    Rather than projecting every point onto one single straight axis (an SVD
    direction, which would iron out any real curvature the line has), this
    builds a smooth "backbone" estimate of the curve itself using a heavier
    gaussian smoothing pass, then measures each point's perpendicular deviation
    from that *local*, curved backbone (using the backbone's own local tangent
    at each point) instead of from a straight line. Only that local deviation
    -- generally small per-point noise -- gets tanh-damped; the overall bend of
    the curve is left alone since it's baked into the backbone itself. A final
    light gaussian pass (spatial_sigma) smooths any remaining roughness.

    Args:
        points: (N, 2) array of (x, y) coordinates for a single frame's centerline.
        scale_limit_px: threshold pixel deviation from the backbone beyond which
            points get compressed.
        spatial_sigma: sigma for the final light gaussian smoothing pass.
        backbone_sigma: sigma (in node-index units) used to build the smooth
            backbone curve deviations are measured against. Larger values
            follow the curve's broad shape only; smaller values track the
            points (and any real curvature) more tightly.
    Returns:
        (N, 2) array of refined (x, y) coordinates. Returned unchanged if there
        are fewer than 5 points (not enough to build a stable backbone).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 5:
        return pts

    backbone_x = gaussian_filter1d(pts[:, 0], sigma=backbone_sigma, mode='nearest')
    backbone_y = gaussian_filter1d(pts[:, 1], sigma=backbone_sigma, mode='nearest')
    backbone = np.column_stack((backbone_x, backbone_y))

    tangents = np.gradient(backbone, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1)
    tangent_norms[tangent_norms < 1e-9] = 1.0
    tangents = tangents / tangent_norms[:, None]

    adjusted_pts = np.zeros_like(pts)
    for idx in range(n):
        pt = pts[idx]
        base_pt = backbone[idx]
        tangent = tangents[idx]

        v = pt - base_pt
        proj_len = np.dot(v, tangent)
        closest_point_on_curve = base_pt + proj_len * tangent

        perp_vector = pt - closest_point_on_curve
        distance = np.linalg.norm(perp_vector)

        if distance > 0.01:
            damped_distance = scale_limit_px * np.tanh(distance / scale_limit_px)
            adjusted_pts[idx] = closest_point_on_curve + (perp_vector / distance) * damped_distance
        else:
            adjusted_pts[idx] = pt

    smoothed_x = gaussian_filter1d(adjusted_pts[:, 0], sigma=spatial_sigma, mode='nearest')
    smoothed_y = gaussian_filter1d(adjusted_pts[:, 1], sigma=spatial_sigma, mode='nearest')

    return np.column_stack((smoothed_x, smoothed_y))


def _windowed_temporal_smooth(values, radius=2, sigma=1.5):
    """
    Smooths a 1D time series using only a hard +/- `radius` frame window around
    each point, instead of scipy's gaussian_filter1d run over the entire series
    (which, despite decaying weights, still processes -- and technically depends
    on -- the full array). Edge frames are padded by replicating the nearest
    real value, similar in spirit to gaussian_filter1d's 'nearest' mode.

    Args:
        values: 1D array of per-frame values for a single node.
        radius: number of neighboring frames on each side to include (e.g. 2
            means each output uses frames [i-2, i+2]).
        sigma: gaussian sigma used to weight samples within the window (closer
            frames count more).
    Returns:
        1D array, same length as `values`, of the windowed-smoothed series.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return values.copy()

    radius = max(1, min(radius, n - 1))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()

    padded = np.pad(values, radius, mode='edge')
    return np.convolve(padded, weights, mode='valid')


def temporal_smooth_window(frame_points_list, center_index, temporal_sigma=1.5):
    """
    Approximates refine_csv's cross-frame temporal smoothing using a small window
    of nearby frames, instead of needing every frame of the video processed first.

    Each entry in frame_points_list must have the same length and the same node
    correspondence (i.e. produced with the same num_nodes), since smoothing is
    applied per node-index across the frame axis, matching how refine_csv
    smooths each node_id across its own local +/- radius frame window.

    Args:
        frame_points_list: ordered list of (N, 2) arrays, one per frame in the
            window (e.g. [frame-2, frame-1, frame, frame+1, frame+2]).
        center_index: index within frame_points_list of the frame to return
            after smoothing (e.g. 2 for the "frame" entry in the example above).
        temporal_sigma: sigma for the gaussian weighting across the frame axis.
    Returns:
        (N, 2) array: the temporally-smoothed points for the requested frame.
    """
    stack = np.stack(frame_points_list, axis=0)  # (F, N, 2)
    num_frames = stack.shape[0]
    weights = np.array([
        np.exp(-0.5 * ((i - center_index) / temporal_sigma) ** 2) for i in range(num_frames)
    ])
    weights /= weights.sum()
    smoothed_x = np.tensordot(weights, stack[:, :, 0], axes=(0, 0))
    smoothed_y = np.tensordot(weights, stack[:, :, 1], axes=(0, 0))
    return np.column_stack((smoothed_x, smoothed_y))


def refine_csv(file_path, scale_limit_px=3.0, spatial_sigma=1.2, temporal_sigma=1.5,
                backbone_sigma=15.0, temporal_radius=2):
    """
    Modifies the coordinates in the csv file as follows:
    1. Uses tanh to compress pixels that are > 3.0px from the local curve backbone in a single frame.
    2. Smooths each node across only its immediate +/- temporal_radius neighboring
       frames (not the whole video's worth of frames) to drop small jumps caused by
       imperfect masks, while still tracking real motion over the course of the video.
    Args:
        file_path: path to the csv file
        scale_limit_px: threshold pixel value (anything above this from the backbone will get damped)
        spatial_sigma: sigma value for gaussian filtering used in step 1
        temporal_sigma: gaussian sigma used to weight frames within the temporal_radius window in step 2
        backbone_sigma: sigma used to build the per-frame backbone curve in step 1
        temporal_radius: number of neighboring frames on each side used for step 2 (e.g. 2 = a 5-frame window)
    No return value.
    """
    df = pd.read_csv(file_path)
    refined_rows = []
    
    for frame_id, group in df.groupby('frame_id'):
        sorted_nodes = group.sort_values('node_id').copy()
        pts = sorted_nodes[['x_pixel', 'y_pixel']].to_numpy()
        
        if len(pts) < 5:
            refined_rows.append(sorted_nodes)
            continue

        refined_pts = refine_frame_points(
            pts, scale_limit_px=scale_limit_px, spatial_sigma=spatial_sigma, backbone_sigma=backbone_sigma
        )

        sorted_nodes['x_pixel'] = refined_pts[:, 0]
        sorted_nodes['y_pixel'] = refined_pts[:, 1]
        refined_rows.append(sorted_nodes)
        
    spatial_df = pd.concat(refined_rows)
    
    temporal_rows = []
    for node_id, group in spatial_df.groupby('node_id'):
        sorted_time = group.sort_values('frame_id').copy()
        
        if len(sorted_time) > 3:
            sorted_time['x_pixel'] = _windowed_temporal_smooth(
                sorted_time['x_pixel'].to_numpy(), radius=temporal_radius, sigma=temporal_sigma
            )
            sorted_time['y_pixel'] = _windowed_temporal_smooth(
                sorted_time['y_pixel'].to_numpy(), radius=temporal_radius, sigma=temporal_sigma
            )
            
        temporal_rows.append(sorted_time)
        
    final_df = pd.concat(temporal_rows).sort_values(['frame_id', 'node_id'])
    final_df.to_csv(file_path, index=False)
    

def create_csv(folders, boundary, input_path, output_path):
    """
    Writes a csv file that is (N * num_nodes) x 5, each row containing 
    frame number, node id, x coordinate, y coordinate, and the global node spacing.
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
        csv_writer.writerow(['frame_id', 'node_id', 'x_pixel', 'y_pixel', 'node_spacing'])

        prev_orientation = None  # carried across frames for hysteresis, so
                                  # orientation doesn't flip frame-to-frame
                                  # right around an aspect ratio of 1
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

            upper_pts, u_min, u_max, lower_pts, l_min, l_max, axis = extract_boundary_pair(
                mask_u, mask_l, prev_orientation=prev_orientation
            )
            if axis is not None:
                prev_orientation = 'vertical' if axis.direction[1] != 0 else 'horizontal'

            skeleton_nodes = centerline(upper_pts, u_min, u_max, lower_pts, l_min, l_max,
                                         num_nodes=2000, trig_harmonics=3, axis=axis)
            
            if skeleton_nodes is not None:
                dx = np.diff(skeleton_nodes[:, 0])
                dy = np.diff(skeleton_nodes[:, 1])
                step_distances = np.sqrt(dx**2 + dy**2)
                node_spacing = np.mean(step_distances)
                
                for node_idx, (x_val, y_val) in enumerate(skeleton_nodes):
                    safe_x = np.clip(x_val, 0, width - 1)
                    safe_y = np.clip(y_val, 0, img_height - 1)
                    csv_writer.writerow([frame_idx, node_idx, f"{safe_x:.4f}", f"{safe_y:.4f}", f"{node_spacing:.6f}"])


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
            refine_csv(output_csv_path, scale_limit_px=3.0, spatial_sigma=1.2, temporal_sigma=1.5)
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
