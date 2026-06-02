import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
from tools.buffer_trajectory import TrajectoryBuffer


STAGE_NAMES = {
    1: "Pick Middle",
    2: "Insert Middle",
    3: "Pick Bottom",
    4: "Insert Bottom",
    5: "Tighten Parts",
}

# --------- analysis config (edit if your keys differ) ----------
NO_TEACHER_KEY_CANDIDATES = [
    ("no_teacher_action",),          # top-level
    ("info", "no_teacher_action"),   # sometimes nested
]
# --------------------------------------------------------------


def _extract_image(step, img_key="image2", bgr=True):
    img = step["obs"][img_key]

    if img.ndim == 4 and img.shape[0] >= 1:
        img = img[0]

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    if bgr and (img.ndim == 3 and img.shape[-1] == 3):
        img = img[..., ::-1]

    if img.min() < 0:
        img = (img + 1.0) / 2.0

    return np.clip(img, 0.0, 1.0)


def _extract_gripper_value(step, gripper_key=("obs", "robot0_eef_pos_vel"), gripper_index=-1):
    x = step
    for k in gripper_key:
        x = x[k]
    x = np.asarray(x).squeeze()

    if x.ndim == 0:
        return float(x)

    if not (-len(x) <= gripper_index < len(x)):
        raise IndexError(
            f"gripper_index={gripper_index} out of bounds for gripper vector shape {x.shape}"
        )

    return float(x[gripper_index])


def _binarize_gripper_signal(gripper_vals, method="median"):
    g = np.asarray(gripper_vals, dtype=np.float32)
    thr = 0.0 if method == "zero" else float(np.median(g))

    # gripper_width: smaller => closed
    closed = (g <= thr).astype(np.int32)
    return closed, thr


def _segment_from_switches(binary_gripper, include_ends=True):
    b = np.asarray(binary_gripper, dtype=np.int32)
    T = len(b)
    switch_points = list(np.where(b[1:] != b[:-1])[0] + 1)

    seg_points = set(switch_points)
    if include_ends:
        seg_points.add(0)
        seg_points.add(T - 1)

    return sorted(seg_points)


def _segments_from_points(points, T):
    """
    Points define boundaries. Segments are between neighbouring points:
      points: [p0, p1, p2, ...]
      segments:
        [p0, p1-1], [p1, p2-1], ... , [plast, T-1]
    """
    pts = sorted(set(int(p) for p in points if 0 <= int(p) < T))
    if 0 not in pts:
        pts = [0] + pts
    if (T - 1) not in pts:
        pts = pts + [T - 1]
    pts = sorted(set(pts))

    segs = []
    for i in range(len(pts) - 1):
        s = pts[i]
        e = pts[i + 1] - 1
        if e >= s:
            segs.append((s, e))

    last = pts[-1]
    if last <= T - 1:
        segs.append((last, T - 1))

    return segs


def _find_segment_index(segments, t):
    for i, (s, e) in enumerate(segments):
        if s <= t <= e:
            return i
    return None


def label_trajectory_stages(
    traj,
    traj_id=0,
    img_key="image2",
    gripper_key=("obs", "robot0_eef_pos_vel"),
    gripper_index=-1,
    bgr=True,
    remove_tol=10,
    gripper_binarize="median",
    initial_seg_points=None,
    initial_segment_labels=None,
):
    T = len(traj)
    timesteps = np.arange(T)

    gripper_vals = np.array(
        [_extract_gripper_value(step, gripper_key, gripper_index) for step in traj],
        dtype=np.float32
    )
    gripper_bin, thr = _binarize_gripper_signal(gripper_vals, method=gripper_binarize)

    auto_points = _segment_from_switches(gripper_bin, include_ends=True)

    if initial_seg_points is None:
        seg_points_init = list(auto_points)
    else:
        seg_points_init = sorted(set(int(p) for p in initial_seg_points if 0 <= int(p) < T))
        if 0 not in seg_points_init:
            seg_points_init = [0] + seg_points_init
        if (T - 1) not in seg_points_init:
            seg_points_init = seg_points_init + [T - 1]
        seg_points_init = sorted(set(seg_points_init))

    state = {
        "seg_points": seg_points_init,
        "segments": _segments_from_points(seg_points_init, T),
        "labels": None,
        "seg_vlines": [],
        "label_spans": [],
        "label_texts": [],
    }

    state["labels"] = [0] * len(state["segments"])
    if initial_segment_labels is not None and len(initial_segment_labels) == len(state["labels"]):
        state["labels"] = [int(x) for x in initial_segment_labels]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 3.6, 0.25])
    ax_plot = fig.add_subplot(gs[0])
    ax_img = fig.add_subplot(gs[1])
    ax_slider = fig.add_subplot(gs[2])

    ax_plot.step(timesteps, gripper_bin, where="post", label="gripper_bin (closed=1)", alpha=0.9)
    ax_plot.plot(timesteps, gripper_vals, label="gripper_width", alpha=0.35)
    ax_plot.set_xlim(0, T - 1)
    ax_plot.set_ylim(min(gripper_vals.min(), -0.2), max(gripper_vals.max(), 1.2))
    ax_plot.set_ylabel("Gripper")
    ax_plot.grid(True, alpha=0.3)

    help_text = (
        "Controls: Left/Right=step | a=add seg point | r=remove near seg point | "
        "1-5=label segment | 0=clear label | Close=save & next\n"
        f"Stages: 1:{STAGE_NAMES[1]}, 2:{STAGE_NAMES[2]}, 3:{STAGE_NAMES[3]}, 4:{STAGE_NAMES[4]}, 5:{STAGE_NAMES[5]}"
    )
    ax_plot.set_title(f"Trajectory {traj_id} — Segmentation + Stage Labeling\n{help_text}", fontsize=10)

    cursor_line = ax_plot.axvline(0, color="k", linestyle="-", alpha=0.8)

    img_display = ax_img.imshow(np.zeros((60, 80, 3), dtype=np.float32))
    ax_img.axis("off")
    img_title = ax_img.set_title("t = 0", fontsize=12)

    slider = Slider(ax=ax_slider, label="Timestep", valmin=0, valmax=T - 1, valinit=0, valstep=1)

    def build_segments_from_state():
        return _segments_from_points(state["seg_points"], T)

    def redraw_segmentation_markers():
        for ln in state["seg_vlines"]:
            ln.remove()
        state["seg_vlines"].clear()

        for p in state["seg_points"]:
            ln = ax_plot.axvline(p, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.5)
            state["seg_vlines"].append(ln)

    def redraw_labels():
        for sp in state["label_spans"]:
            sp.remove()
        for tx in state["label_texts"]:
            tx.remove()
        state["label_spans"].clear()
        state["label_texts"].clear()

        old_segments = state["segments"]
        old_labels = state["labels"]

        new_segments = build_segments_from_state()
        new_labels = [0] * len(new_segments)

        # preserve labels by mapping midpoints
        for i, (s, e) in enumerate(new_segments):
            mid = (s + e) // 2
            old_i = _find_segment_index(old_segments, mid)
            if old_i is not None and old_i < len(old_labels):
                new_labels[i] = int(old_labels[old_i])

        state["segments"] = new_segments
        state["labels"] = new_labels

        for (s, e), lab in zip(state["segments"], state["labels"]):
            lab = int(lab)
            if lab == 0:
                continue

            sp = ax_plot.axvspan(s, e, color="lightgreen", alpha=0.25)
            state["label_spans"].append(sp)

            cx = (s + e) / 2.0
            cy = ax_plot.get_ylim()[1] * 0.85
            tx = ax_plot.text(
                cx, cy, f"{lab}",
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none"),
            )
            state["label_texts"].append(tx)

        fig.canvas.draw_idle()

    def update(_):
        t = int(slider.val)
        cursor_line.set_xdata([t, t])

        img = _extract_image(traj[t], img_key=img_key, bgr=bgr)
        img_display.set_data(img)

        seg_i = _find_segment_index(state["segments"], t)
        if seg_i is None:
            seg_info = "seg: ?"
        else:
            s, e = state["segments"][seg_i]
            lab = int(state["labels"][seg_i])
            lab_name = STAGE_NAMES.get(lab, "Unlabeled") if lab != 0 else "Unlabeled"
            seg_info = f"seg[{seg_i}]={s}..{e} | label={lab} ({lab_name})"

        img_title.set_text(
            f"t={t} | gripper_width={gripper_vals[t]:.4f} | closed={gripper_bin[t]} | {seg_info}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def add_seg_point(t):
        p = int(t)
        if p <= 0 or p >= T - 1:
            print(f"[traj {traj_id}] Not adding point at boundary t={p}.")
            return
        if p in state["seg_points"]:
            print(f"[traj {traj_id}] Point t={p} already exists.")
            return
        state["seg_points"] = sorted(set(state["seg_points"] + [p]))
        print(f"[traj {traj_id}] Added segmentation point: {p}")
        redraw_segmentation_markers()
        redraw_labels()

    def remove_seg_point(t):
        p = int(t)
        candidates = [q for q in state["seg_points"] if q not in (0, T - 1)]
        if not candidates:
            print(f"[traj {traj_id}] No removable segmentation points.")
            return
        nearest = min(candidates, key=lambda q: abs(q - p))
        if abs(nearest - p) <= int(remove_tol):
            state["seg_points"] = sorted([q for q in state["seg_points"] if q != nearest])
            print(f"[traj {traj_id}] Removed segmentation point: {nearest}")
            redraw_segmentation_markers()
            redraw_labels()
        else:
            print(f"[traj {traj_id}] No seg point within ±{remove_tol} of t={p} (nearest={nearest}).")

    def set_label_for_current_segment(t, label):
        label = int(label)  # 0..5
        seg_i = _find_segment_index(state["segments"], int(t))
        if seg_i is None:
            print(f"[traj {traj_id}] Could not find segment for t={t}")
            return

        state["labels"][seg_i] = label
        s, e = state["segments"][seg_i]
        if label == 0:
            print(f"[traj {traj_id}] Cleared label for segment {seg_i} ({s}..{e})")
        else:
            print(f"[traj {traj_id}] Labeled segment {seg_i} ({s}..{e}) as {label}: {STAGE_NAMES[label]}")
        redraw_labels()

    def on_key(event):
        t = int(slider.val)

        if event.key == "right":
            slider.set_val(min(t + 1, T - 1))
        elif event.key == "left":
            slider.set_val(max(t - 1, 0))
        elif event.key == "a":
            add_seg_point(t)
            update(t)
        elif event.key == "r":
            remove_seg_point(t)
            update(t)
        elif event.key in ["1", "2", "3", "4", "5"]:
            set_label_for_current_segment(t, int(event.key))
            update(t)
        elif event.key == "0":
            set_label_for_current_segment(t, 0)
            update(t)

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw_segmentation_markers()
    redraw_labels()
    update(0)
    plt.show()

    out_segments = []
    for (s, e), lab in zip(state["segments"], state["labels"]):
        out_segments.append({"start": int(s), "end": int(e), "label": int(lab)})

    return {
        "seg_points": [int(x) for x in state["seg_points"]],
        "segments": out_segments,
        "meta": {
            "traj_id": int(traj_id),
            "T": int(T),
            "img_key": str(img_key),
            "gripper_key": list(gripper_key),
            "gripper_index": int(gripper_index),
            "binarize_method": gripper_binarize,
            "threshold_used": float(thr),
            "label_map": {str(k): v for k, v in STAGE_NAMES.items()},
            "remove_tol": int(remove_tol),
        },
    }


# =========================
# NEW: ANALYSIS FUNCTIONS
# =========================
def _get_nested(step, path):
    cur = step
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _extract_no_teacher(step):
    """
    Return 1 if no teacher action, else 0.
    Tries a few candidate key paths.
    """
    for path in NO_TEACHER_KEY_CANDIDATES:
        v = _get_nested(step, path)
        if v is not None:
            return int(v)
    raise KeyError(
        f"Could not find no_teacher_action in step. Tried: {NO_TEACHER_KEY_CANDIDATES}"
    )


def build_step_labels_from_segments(T, segments_list):
    """
    segments_list: [{"start":s,"end":e,"label":lab}, ...]
    Returns per-step label array of length T (int, default 0).
    """
    step_labels = np.zeros(T, dtype=np.int32)
    for seg in segments_list:
        s = int(seg["start"])
        e = int(seg["end"])
        lab = int(seg.get("label", 0))
        s = max(0, min(T - 1, s))
        e = max(0, min(T - 1, e))
        if e >= s:
            step_labels[s:e + 1] = lab
    return step_labels


def count_teacher_windows_by_label(traj, step_labels, min_len=16):
    """
    For each step t:
      if teacher action is available at t and for next min_len-1 steps
      -> count label at step t.
    Returns dict {label:int_count} including label 0.
    """
    T = len(traj)
    no_teacher = np.array([_extract_no_teacher(traj[t]) for t in range(T)], dtype=np.int32)
    teacher_avail = (1 - no_teacher).astype(np.int32)  # 1 if teacher available

    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    if T < min_len:
        return counts

    # sliding window sum to test consecutive availability
    window = np.ones(min_len, dtype=np.int32)
    conv = np.convolve(teacher_avail, window, mode="valid")  # length T-min_len+1

    valid_starts = (conv == min_len)  # teacher available for [t, t+min_len-1]

    for t in np.where(valid_starts)[0]:
        lab = int(step_labels[t])
        if lab not in counts:
            counts[lab] = 0
        counts[lab] += 1

    return counts


def analyse_dataset_with_labels(buffer_path, labels_json_path, min_len=16, print_per_traj=True):
    """
    Loads labeled segments from json, then for each traj:
      counts valid teacher windows by label.
    Prints per-traj summary and dataset summary.
    Returns (per_traj_counts, total_counts).
    """
    with open(labels_json_path, "r") as f:
        all_results = json.load(f)

    traj_buffer = TrajectoryBuffer()

    total_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    per_traj_counts = {}

    traj_ids = sorted(int(k) for k in all_results.keys())

    for traj_id in traj_ids:
        traj = traj_buffer.load_from_file(buffer_path, traj_id=traj_id)
        T = len(traj)

        segments_list = all_results[str(traj_id)].get("segments", [])
        step_labels = build_step_labels_from_segments(T, segments_list)

        counts = count_teacher_windows_by_label(traj, step_labels, min_len=min_len)
        per_traj_counts[traj_id] = counts

        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + int(v)

        if print_per_traj:
            pretty = " | ".join(
                [f"{lab}:{counts.get(lab,0)}" for lab in [0,1,2,3,4,5]]
            )
            print(f"[traj {traj_id}] valid_teacher_windows_by_label => {pretty}")

    print("\n=== DATASET SUMMARY (valid teacher windows counted at start step) ===")
    for lab in [0, 1, 2, 3, 4, 5]:
        name = "Unlabeled" if lab == 0 else STAGE_NAMES[lab]
        print(f"Label {lab:>1} ({name:<13}): {total_counts.get(lab,0)}")

    return per_traj_counts, total_counts


def main(buffer_path):
    # --------- USER CONFIG ---------
    img_key = "image2"
    gripper_key = ("obs", "robot0_eef_pos_vel")
    gripper_index = -1
    start_traj = 0
    # --------------------------------

    traj_buffer = TrajectoryBuffer()
    traj_number = traj_buffer.count_trajectories_in_hdf5(buffer_path)
    print("traj_number:", traj_number)

    parent_dir = os.path.dirname(buffer_path)
    filename_no_ext = os.path.splitext(os.path.basename(buffer_path))[0]
    json_filename = f"labeled_stages_{filename_no_ext}.json"
    json_output_path = os.path.join(parent_dir, json_filename)
    print(f"Labeling output will be saved to:\n{json_output_path}\n")

    # resume if exists
    all_results = {}
    if os.path.exists(json_output_path):
        try:
            with open(json_output_path, "r") as f:
                all_results = json.load(f)
            print(f"Loaded existing labels with {len(all_results)} trajectories from {json_output_path}")
        except Exception as e:
            print(f"Warning: could not load existing json ({e}). Starting fresh.")
            all_results = {}

    for traj_id in range(start_traj, traj_number):
        print(f"\n=== Processing Trajectory {traj_id} / {traj_number-1} ===")
        traj = traj_buffer.load_from_file(buffer_path, traj_id=traj_id)

        init_seg_points = None
        init_seg_labels = None
        if str(traj_id) in all_results:
            prev = all_results[str(traj_id)]
            init_seg_points = prev.get("seg_points", None)
            if "segments" in prev and isinstance(prev["segments"], list):
                init_seg_labels = [int(s.get("label", 0)) for s in prev["segments"]]

        result = label_trajectory_stages(
            traj,
            traj_id=traj_id,
            img_key=img_key,
            gripper_key=gripper_key,
            gripper_index=gripper_index,
            bgr=True,
            remove_tol=10,
            gripper_binarize="median",
            initial_seg_points=init_seg_points,
            initial_segment_labels=init_seg_labels,
        )

        all_results[str(traj_id)] = result
        with open(json_output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f" -> Saved traj {traj_id} to {json_output_path}")

    print("\nAll trajectories labeled.\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label and analyze trajectory stages in an HDF5 trajectory buffer."
    )
    parser.add_argument("buffer_path", help="Path to the trajectory HDF5 buffer.")
    args = parser.parse_args()
    buffer_path = os.path.expanduser(os.path.expandvars(args.buffer_path))

    parent_dir = os.path.dirname(buffer_path)
    filename_no_ext = os.path.splitext(os.path.basename(buffer_path))[0]
    json_filename = f"labeled_stages_{filename_no_ext}.json"
    # json_filename = "labeled_stages_trajectory_buffer_intervention_nov21_saved.json"
    json_output_path = os.path.join(parent_dir, json_filename)

    main(buffer_path)  # (optional) label first; comment out if already labeled

    analyse_dataset_with_labels(
        buffer_path=buffer_path,
        labels_json_path=json_output_path,
        min_len=16,
        print_per_traj=True
    )
