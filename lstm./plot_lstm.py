#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recover and plot saved LSTM temporal-window results.

This script DOES NOT retrain anything.

It loads the existing *_results.npz files and reconstructs:
    - mean accuracy
    - SEM
    - participant accuracies
    - group confusion matrices
    - normalized confusion matrices
    - group node importance
    - temporal decoding plot

Currently available:
    0–100 ms
    100–200 ms
    200–300 ms
    300–400 ms
    400–500 ms
    500–600 ms
    600–700 ms
    700–800 ms
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = Path(
    "/home/uranus/Scrivania/MEG_replay/epoched_data/"
    "lstm_temporal_models"
)

CHANCE = 0.25


# =========================================================
# FIND RESULT FILES
# =========================================================

result_files = sorted(
    MODEL_PATH.glob("*_results.npz")
)

print("\n======================================")
print("RECOVERING SAVED LSTM RESULTS")
print("======================================")

print(
    f"Result files found: {len(result_files)}"
)

for f in result_files:
    print(" ", f.name)


# =========================================================
# STORAGE
# =========================================================

windows = []
mean_accuracy = []
sem_accuracy = []

participant_results = {}

confusion_matrices = {}
normalized_confusion_matrices = {}
node_importance = {}


# =========================================================
# LOAD EACH WINDOW
# =========================================================

for result_file in result_files:

    print(
        f"\nLoading: {result_file.name}"
    )

    data = np.load(
        result_file,
        allow_pickle=True
    )

    # -----------------------------------------------------
    # Window
    # -----------------------------------------------------

    start = float(
        data["window_start"]
    )

    end = float(
        data["window_end"]
    )

    label = (
        f"{int(start * 1000):04d}_"
        f"{int(end * 1000):04d}ms"
    )

    windows.append(label)

    # -----------------------------------------------------
    # Participant accuracy
    # -----------------------------------------------------

    acc = data[
        "participant_accuracy"
    ]

    participant_results[label] = acc

    # -----------------------------------------------------
    # Calculate mean + SEM ourselves
    # -----------------------------------------------------

    mean = np.mean(acc)

    if len(acc) > 1:

        sem = (
            np.std(
                acc,
                ddof=1
            )
            /
            np.sqrt(
                len(acc)
            )
        )

    else:

        sem = np.nan

    mean_accuracy.append(mean)
    sem_accuracy.append(sem)

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    if "group_confusion_matrix" in data:

        confusion_matrices[label] = data[
            "group_confusion_matrix"
        ]

    if "group_confusion_matrix_normalized" in data:

        normalized_confusion_matrices[label] = data[
            "group_confusion_matrix_normalized"
        ]

    # -----------------------------------------------------
    # Node importance
    # -----------------------------------------------------

    if "group_node_importance" in data:

        node_importance[label] = data[
            "group_node_importance"
        ]

    # -----------------------------------------------------
    # Print
    # -----------------------------------------------------

    print(
        f"  Participants: {len(acc)}"
    )

    print(
        f"  Mean accuracy: {mean:.3f}"
    )

    print(
        f"  SEM: {sem:.3f}"
    )


# =========================================================
# SORT BY TEMPORAL ORDER
# =========================================================

sort_idx = np.argsort(
    [
        int(
            w.split("_")[0]
        )
        for w in windows
    ]
)

windows = [
    windows[i]
    for i in sort_idx
]

mean_accuracy = np.array(
    [
        mean_accuracy[i]
        for i in sort_idx
    ]
)

sem_accuracy = np.array(
    [
        sem_accuracy[i]
        for i in sort_idx
    ]
)


# =========================================================
# PRINT SUMMARY
# =========================================================

print(
    "\n======================================"
)

print(
    "RECOVERED RESULTS"
)

print(
    "======================================"
)

for w, mean, sem in zip(
    windows,
    mean_accuracy,
    sem_accuracy
):

    print(
        f"{w:15s} "
        f"{mean:.3f} ± {sem:.3f}"
    )


# =========================================================
# BEST WINDOW
# =========================================================

best_idx = np.argmax(
    mean_accuracy
)

print(
    "\n======================================"
)

print(
    "BEST SAVED WINDOW"
)

print(
    "======================================"
)

print(
    f"Window: {windows[best_idx]}"
)

print(
    f"Accuracy: "
    f"{mean_accuracy[best_idx]:.3f}"
)

print(
    f"SEM: "
    f"{sem_accuracy[best_idx]:.3f}"
)


# =========================================================
# TEMPORAL DECODING PLOT
# =========================================================

x = np.arange(
    len(windows)
)

plt.figure(
    figsize=(13, 6)
)

plt.errorbar(
    x,
    mean_accuracy,
    yerr=sem_accuracy,
    marker="o",
    linewidth=2,
    capsize=4
)

plt.axhline(
    CHANCE,
    linestyle="--",
    linewidth=2,
    label="Chance (25%)"
)

plt.xticks(
    x,
    windows,
    rotation=45
)

plt.xlabel(
    "Temporal window"
)

plt.ylabel(
    "Mean test accuracy"
)

plt.title(
    "4-class LSTM decoding across temporal windows"
)

plt.legend()

plt.tight_layout()


plot_path = MODEL_PATH / (
    "RECOVERED_temporal_window_decoding_accuracy.png"
)

plt.savefig(
    plot_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# SAVE RECOVERED SUMMARY
# =========================================================

summary_path = MODEL_PATH / (
    "RECOVERED_temporal_window_summary.npz"
)

np.savez(
    summary_path,

    window_labels=np.array(
        windows
    ),

    mean_accuracy=mean_accuracy,

    sem_accuracy=sem_accuracy,

    chance=CHANCE
)


print(
    "\n======================================"
)

print(
    "DONE"
)

print(
    "======================================"
)

print(
    f"Plot saved to:\n{plot_path}"
)

print(
    f"\nSummary saved to:\n{summary_path}"
)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detailed inspection of the 500–600 ms LSTM decoding window.

Loads already-saved results.
NO retraining.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn import plotting

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = Path(
    "/home/uranus/Scrivania/MEG_replay/epoched_data/"
    "lstm_temporal_models"
)

RESULT_FILE = MODEL_PATH / (
    "0300_0400ms_results.npz"
)

CONDITION_NAMES = [
    "Stay Left",
    "Stay Right",
    "Shift Left",
    "Shift Right"
]

CHANCE = 0.25


# =========================================================
# LOAD
# =========================================================

data = np.load(
    RESULT_FILE,
    allow_pickle=True
)

print("\n======================================")
print("500–600 ms WINDOW")
print("======================================")

print("\nVariables saved in file:")

for key in data.files:
    print(
        f"{key:40s}",
        data[key].shape
    )


# =========================================================
# EXTRACT
# =========================================================

participant_accuracy = data[
    "participant_accuracy"
]

participant_names = data[
    "participant_names"
]

group_cm = data[
    "group_confusion_matrix"
]

group_cm_norm = data[
    "group_confusion_matrix_normalized"
]

node_importance = data[
    "group_node_importance"
]

classifier_weights = data[
    "classifier_weights"
]

classifier_bias = data[
    "classifier_bias"
]

lstm_kernel = data[
    "lstm_kernel"
]

lstm_recurrent_kernel = data[
    "lstm_recurrent_kernel"
]

lstm_bias = data[
    "lstm_bias"
]

scaler_means = data[
    "scaler_means"
]

scaler_scales = data[
    "scaler_scales"
]


# =========================================================
# BASIC STATISTICS
# =========================================================

mean_accuracy = np.mean(
    participant_accuracy
)

sem_accuracy = (
    np.std(
        participant_accuracy,
        ddof=1
    )
    /
    np.sqrt(
        len(participant_accuracy)
    )
)

print("\n======================================")
print("ACCURACY")
print("======================================")

print(
    f"N = {len(participant_accuracy)}"
)

print(
    f"Mean = {mean_accuracy:.4f}"
)

print(
    f"SEM = {sem_accuracy:.4f}"
)

print(
    f"Chance = {CHANCE:.4f}"
)


# =========================================================
# PARTICIPANT ACCURACY
# =========================================================

plt.figure(
    figsize=(12, 5)
)

x = np.arange(
    len(participant_accuracy)
)

plt.bar(
    x,
    participant_accuracy
)

plt.axhline(
    CHANCE,
    linestyle="--",
    linewidth=2,
    label="Chance (25%)"
)

plt.xticks(
    x,
    participant_names,
    rotation=45
)

plt.ylabel(
    "Test accuracy"
)

plt.xlabel(
    "Participant"
)

plt.title(
    "500–600 ms: Participant decoding accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_participant_accuracy.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# GROUP ACCURACY
# =========================================================

plt.figure(
    figsize=(5, 6)
)

plt.errorbar(
    [0],
    [mean_accuracy],
    yerr=[sem_accuracy],
    fmt="o",
    markersize=10,
    capsize=6,
    linewidth=2
)

plt.axhline(
    CHANCE,
    linestyle="--",
    linewidth=2,
    label="Chance"
)

plt.xlim(
    -0.5,
    0.5
)

plt.xticks(
    [0],
    ["300–400 ms"]
)

plt.ylabel(
    "Mean test accuracy"
)

plt.title(
    "300–400 ms decoding"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_group_accuracy.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    group_cm
)

plt.colorbar(
    label="Number of trials"
)

plt.xticks(
    range(4),
    CONDITION_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(4),
    CONDITION_NAMES
)

plt.xlabel(
    "Predicted condition"
)

plt.ylabel(
    "True condition"
)

plt.title(
    "300–400 ms: Group confusion matrix"
)

for i in range(4):

    for j in range(4):

        plt.text(
            j,
            i,
            str(group_cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_confusion_matrix.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# NORMALIZED CONFUSION MATRIX
# =========================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    group_cm_norm,
    vmin=0,
    vmax=1
)

plt.colorbar(
    label="Proportion"
)

plt.xticks(
    range(4),
    CONDITION_NAMES,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(4),
    CONDITION_NAMES
)

plt.xlabel(
    "Predicted condition"
)

plt.ylabel(
    "True condition"
)

plt.title(
    "300–400 ms: Normalized confusion matrix"
)

for i in range(4):

    for j in range(4):

        plt.text(
            j,
            i,
            f"{group_cm_norm[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_normalized_confusion_matrix.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# NODE IMPORTANCE
# =========================================================

print("\n======================================")
print("NODE IMPORTANCE")
print("======================================")

print(
    "Shape:",
    node_importance.shape
)


# ---------------------------------------------------------
# Plot all four conditions
# ---------------------------------------------------------

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 12),
    sharex=True
)

for class_number in range(4):

    values = node_importance[
        class_number
    ]

    axes[class_number].bar(
        np.arange(
            len(values)
        ),
        values
    )

    axes[class_number].axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    axes[class_number].set_ylabel(
        CONDITION_NAMES[class_number]
    )

axes[-1].set_xlabel(
    "MEG node"
)

fig.suptitle(
    "300–400 ms: Gradient × Input node importance",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_node_importance_all_conditions.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()

# =========================================================
# GLASS BRAIN: NODE IMPORTANCE
# =========================================================

print("\n======================================")
print("GLASS BRAIN NODE IMPORTANCE")
print("======================================")


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

COORDINATE_FILE = Path(
    "/home/uranus/Scrivania/MEG_replay/"
    "coordinates_ROIS_fMRI/"
    "config_net_labels_after_VIVI_DEF_MAU_last.xlsx"
)

# Plot only the highest-importance nodes
TOP_PERCENT = 10

# Minimum number of nodes to display
MIN_NODES = 5


# ---------------------------------------------------------
# LOAD COORDINATES
# ---------------------------------------------------------

import pandas as pd

coords_df = pd.read_excel(
    COORDINATE_FILE
)

print("\nCoordinate file:")
print(COORDINATE_FILE)

print("\nColumns found:")
print(coords_df.columns.tolist())

print("\nCoordinate table shape:")
print(coords_df.shape)

print(coords_df.head())


# ---------------------------------------------------------
# FIND X/Y/Z COLUMNS
# ---------------------------------------------------------

possible_x = [
    "x", "X",
    "MNI_x", "MNI X",
    "MNI_X",
    "coord_x", "Coord_X"
]

possible_y = [
    "y", "Y",
    "MNI_y", "MNI Y",
    "MNI_Y",
    "coord_y", "Coord_Y"
]

possible_z = [
    "z", "Z",
    "MNI_z", "MNI Z",
    "MNI_Z",
    "coord_z", "Coord_Z"
]


def find_column(
    dataframe,
    possible_names
):

    for name in possible_names:

        if name in dataframe.columns:
            return name

    return None


x_col = find_column(
    coords_df,
    possible_x
)

y_col = find_column(
    coords_df,
    possible_y
)

z_col = find_column(
    coords_df,
    possible_z
)


print("\nDetected coordinate columns:")

print("X:", x_col)
print("Y:", y_col)
print("Z:", z_col)


if (
    x_col is None
    or y_col is None
    or z_col is None
):

    raise ValueError(
        "\nCould not automatically find MNI X/Y/Z columns.\n"
        "Please check the printed column names above."
    )


# ---------------------------------------------------------
# EXTRACT MNI COORDINATES
# ---------------------------------------------------------

mni_coords = coords_df[
    [
        x_col,
        y_col,
        z_col
    ]
].to_numpy(
    dtype=float
)


print(
    "\nMNI coordinate shape:",
    mni_coords.shape
)

print(
    "Node importance shape:",
    node_importance.shape
)


# ---------------------------------------------------------
# CHECK NUMBER OF NODES
# ---------------------------------------------------------

n_nodes = node_importance.shape[-1]

if len(mni_coords) != n_nodes:

    raise ValueError(
        f"\nNumber of coordinates ({len(mni_coords)}) "
        f"does not match number of MEG nodes ({n_nodes}).\n\n"
        "The coordinate file must have exactly one "
        "MNI coordinate triplet for each MEG node."
    )


# =========================================================
# FUNCTION: CREATE NIFTI FROM NODE VALUES
# =========================================================

def node_values_to_nifti(
    values,
    coords,
    threshold_percent=10
):

    values = np.asarray(values)

    # -----------------------------------------------------
    # Keep only strongest nodes
    # -----------------------------------------------------

    absolute_values = np.abs(values)

    threshold = np.percentile(
        absolute_values,
        100 - threshold_percent
    )

    significant_mask = (
        absolute_values >= threshold
    )

    # Ensure minimum number of nodes
    if np.sum(significant_mask) < MIN_NODES:

        top_indices = np.argsort(
            absolute_values
        )[::-1][:MIN_NODES]

        significant_mask = np.zeros(
            len(values),
            dtype=bool
        )

        significant_mask[
            top_indices
        ] = True

    print(
        f"Displaying "
        f"{np.sum(significant_mask)} / "
        f"{len(values)} nodes"
    )

    # -----------------------------------------------------
    # Create a small MNI-space volume
    #
    # MNI coordinates are converted into a
    # 2 mm voxel grid.
    # -----------------------------------------------------

    voxel_size = 2.0

    # MNI bounding box
    xmin, xmax = -90, 90
    ymin, ymax = -126, 90
    zmin, zmax = -72, 108

    shape = (
        int((xmax - xmin) / voxel_size) + 1,
        int((ymax - ymin) / voxel_size) + 1,
        int((zmax - zmin) / voxel_size) + 1
    )

    volume = np.zeros(
        shape,
        dtype=float
    )

    affine = np.array(
        [
            [voxel_size, 0, 0, xmin],
            [0, voxel_size, 0, ymin],
            [0, 0, voxel_size, zmin],
            [0, 0, 0, 1]
        ]
    )

    # -----------------------------------------------------
    # Put node values into volume
    # -----------------------------------------------------

    for node_idx in np.where(
        significant_mask
    )[0]:

        x, y, z = coords[node_idx]

        voxel_x = int(
            round(
                (x - xmin) / voxel_size
            )
        )

        voxel_y = int(
            round(
                (y - ymin) / voxel_size
            )
        )

        voxel_z = int(
            round(
                (z - zmin) / voxel_size
            )
        )

        if (
            0 <= voxel_x < shape[0]
            and
            0 <= voxel_y < shape[1]
            and
            0 <= voxel_z < shape[2]
        ):

            volume[
                voxel_x,
                voxel_y,
                voxel_z
            ] = values[node_idx]

    return nib.Nifti1Image(
        volume,
        affine
    )


# =========================================================
# GLASS BRAIN FOR EACH CONDITION
# =========================================================

for class_number in range(4):

    values = node_importance[
        class_number
    ]

    print(
        f"\n--------------------------------------"
    )

    print(
        CONDITION_NAMES[class_number]
    )

    print(
        "Min:",
        np.min(values)
    )

    print(
        "Max:",
        np.max(values)
    )

    # -----------------------------------------------------
    # Convert node values to MNI volume
    # -----------------------------------------------------

    nii = node_values_to_nifti(
        values,
        mni_coords,
        threshold_percent=TOP_PERCENT
    )

    # -----------------------------------------------------
    # Glass brain
    # -----------------------------------------------------

    display = plotting.plot_glass_brain(
        nii,
        display_mode="lyrz",
        colorbar=True,
        plot_abs=False,
        threshold=0,
        black_bg=False,
        title=(
            f"300–400 ms: "
            f"{CONDITION_NAMES[class_number]}\n"
            f"Top {TOP_PERCENT}% node importance"
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_file = (
        MODEL_PATH /
        (
            f"300_400ms_glass_brain_"
            f"{CONDITION_NAMES[class_number].replace(' ', '_')}"
            f"_node_importance.png"
        )
    )

    display.savefig(
        output_file,
        dpi=600
    )

    display.close()

    print(
        "Saved:",
        output_file
    )


# =========================================================
# OVERALL NODE IMPORTANCE GLASS BRAIN
# =========================================================

print("\n======================================")
print("OVERALL GLASS BRAIN")
print("======================================")


overall_values = np.mean(
    np.abs(
        node_importance
    ),
    axis=0
)


print(
    "Overall importance shape:",
    overall_values.shape
)

print(
    "Maximum:",
    np.max(overall_values)
)


# ---------------------------------------------------------
# Convert to NIfTI
# ---------------------------------------------------------

overall_nii = node_values_to_nifti(
    overall_values,
    mni_coords,
    threshold_percent=TOP_PERCENT
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

display = plotting.plot_glass_brain(
    overall_nii,
    display_mode="lyrz",
    colorbar=True,
    plot_abs=False,
    threshold=0,
    black_bg=False,
    title=(
        "300–400 ms: Overall node importance\n"
        f"Top {TOP_PERCENT}% nodes"
    )
)


output_file = (
    MODEL_PATH /
    "300_400ms_glass_brain_overall_node_importance.png"
)

display.savefig(
    output_file,
    dpi=600
)

display.close()

print(
    "\nSaved:",
    output_file
)


# =========================================================
# GLASS BRAIN WITH ALL MEG NODE MARKERS
# =========================================================

import pandas as pd
from nilearn import plotting


print("\n======================================")
print("GLASS BRAIN — ALL MEG NODES")
print("======================================")


# =========================================================
# LOAD MNI COORDINATES
# =========================================================

COORDINATE_FILE = Path(
    "/home/uranus/Scrivania/MEG_replay/"
    "coordinates_ROIS_fMRI/"
    "config_net_labels_after_VIVI_DEF_MAU_last.xlsx"
)

coordinates = pd.read_excel(
    COORDINATE_FILE
)


print("\nCoordinate file:")
print(COORDINATE_FILE)

print(
    "\nCoordinate dataframe shape:",
    coordinates.shape
)

print(
    "\nCoordinate columns:"
)

print(
    coordinates.columns.tolist()
)


# =========================================================
# FIND X / Y / Z COLUMNS
# =========================================================

possible_x = [
    "x",
    "X",
    "MNI_x",
    "MNI_X",
    "MNI X",
    "coord_x",
    "Coord_X"
]

possible_y = [
    "y",
    "Y",
    "MNI_y",
    "MNI_Y",
    "MNI Y",
    "coord_y",
    "Coord_Y"
]

possible_z = [
    "z",
    "Z",
    "MNI_z",
    "MNI_Z",
    "MNI Z",
    "coord_z",
    "Coord_Z"
]


def find_column(
    dataframe,
    possible_names
):

    for name in possible_names:

        if name in dataframe.columns:

            return name

    return None


x_col = find_column(
    coordinates,
    possible_x
)

y_col = find_column(
    coordinates,
    possible_y
)

z_col = find_column(
    coordinates,
    possible_z
)


print("\nDetected MNI columns:")

print(
    "X:",
    x_col
)

print(
    "Y:",
    y_col
)

print(
    "Z:",
    z_col
)


# =========================================================
# CHECK
# =========================================================

if (
    x_col is None
    or y_col is None
    or z_col is None
):

    raise ValueError(
        "\nCould not identify MNI X/Y/Z columns.\n\n"
        "Available columns are:\n"
        f"{coordinates.columns.tolist()}"
    )


# =========================================================
# EXTRACT COORDINATES
# =========================================================

mni_coords = coordinates[
    [
        x_col,
        y_col,
        z_col
    ]
].to_numpy(
    dtype=float
)


print(
    "\nMNI coordinate array shape:",
    mni_coords.shape
)

print(
    "Node importance shape:",
    node_importance.shape
)


# =========================================================
# CHECK NUMBER OF NODES
# =========================================================

n_nodes = node_importance.shape[-1]

if len(mni_coords) != n_nodes:

    raise ValueError(
        "\nNumber of coordinates does not match "
        "number of MEG nodes!\n\n"
        f"Coordinates = {len(mni_coords)}\n"
        f"MEG nodes   = {n_nodes}"
    )


print(
    f"\n✓ All {n_nodes} MEG nodes have coordinates."
)


# =========================================================
# MARKER SIZE FUNCTION
# =========================================================

def get_marker_sizes(
    values,
    min_size=25,
    max_size=180
):

    abs_values = np.abs(
        values
    )

    maximum = np.max(
        abs_values
    )

    if maximum == 0:

        return np.full(
            len(values),
            min_size
        )

    normalized = (
        abs_values / maximum
    )

    sizes = (
        min_size
        +
        normalized
        *
        (
            max_size
            -
            min_size
        )
    )

    return sizes


# =========================================================
# GLASS BRAIN FOR EACH CONDITION
# =========================================================

for class_number in range(4):

    values = node_importance[
        class_number
    ]

    print(
        "\n--------------------------------------"
    )

    print(
        CONDITION_NAMES[class_number]
    )

    print(
        "Minimum:",
        np.min(values)
    )

    print(
        "Maximum:",
        np.max(values)
    )


    # -----------------------------------------------------
    # Marker sizes
    # -----------------------------------------------------

    marker_sizes = get_marker_sizes(
        values
    )


    # -----------------------------------------------------
    # Create glass brain
    # -----------------------------------------------------

    display = plotting.plot_glass_brain(
        None,
        display_mode="lyrz",
        black_bg=False,
        colorbar=False,
        title=(
            f"300–400 ms — "
            f"{CONDITION_NAMES[class_number]}\n"
            "All MEG nodes"
        )
    )


    # -----------------------------------------------------
    # ADD ROUND MARKERS
    # -----------------------------------------------------

    display.add_markers(
        marker_coords=mni_coords,
        marker_color=values,
        marker_size=marker_sizes,
        alpha=0.85,
        cmap="coolwarm",
        vmin=-np.max(np.abs(values)),
        vmax=np.max(np.abs(values))
    )


    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    output_file = (
        MODEL_PATH /
        (
            "300_400ms_glass_brain_"
            "ALL_NODES_"
            f"{CONDITION_NAMES[class_number].replace(' ', '_')}.png"
        )
    )

    display.savefig(
        output_file,
        dpi=600
    )

    display.close()

    print(
        "Saved:",
        output_file
    )


# =========================================================
# OVERALL NODE IMPORTANCE
# =========================================================

print("\n======================================")
print("OVERALL NODE IMPORTANCE")
print("======================================")


overall_values = np.mean(
    np.abs(
        node_importance
    ),
    axis=0
)


overall_sizes = get_marker_sizes(
    overall_values
)


display = plotting.plot_glass_brain(
    None,
    display_mode="lyrz",
    black_bg=False,
    colorbar=False,
    title=(
        "300–400 ms — Overall MEG node importance\n"
        "All nodes"
    )
)


display.add_markers(
    marker_coords=mni_coords,
    marker_color=overall_values,
    marker_size=overall_sizes,
    alpha=0.85,
    cmap="viridis"
)


output_file = (
    MODEL_PATH /
    "300_400ms_glass_brain_ALL_NODES_OVERALL.png"
)


display.savefig(
    output_file,
    dpi=600
)

display.close()


print(
    "Saved:",
    output_file
)


# =========================================================
# FINISHED
# =========================================================

print("\n======================================")
print("GLASS BRAIN COMPLETE")
print("======================================")

print(
    f"Displayed all {n_nodes} MEG nodes."
)
# =========================================================
# TOP NODES
# =========================================================

# =========================================================
# TOP NODE IMPORTANCE WITH LABELS
# =========================================================

print("\n======================================")
print("TOP MEG NODES — LABELS")
print("======================================")


# ---------------------------------------------------------
# Check labels column
# ---------------------------------------------------------

print(
    "\nCoordinate file columns:"
)

print(
    coordinates.columns.tolist()
)


if "labels" not in coordinates.columns:

    raise ValueError(
        "\nCould not find a column named 'labels'.\n"
        "Available columns are:\n"
        f"{coordinates.columns.tolist()}"
    )


labels = coordinates["labels"].astype(str).to_numpy()


# ---------------------------------------------------------
# Check dimensions
# ---------------------------------------------------------

n_nodes = node_importance.shape[-1]

if len(labels) != n_nodes:

    raise ValueError(
        f"\nMismatch between labels and node importance:\n"
        f"Labels: {len(labels)}\n"
        f"Nodes:  {n_nodes}"
    )


# =========================================================
# TOP N
# =========================================================

TOP_N = 20


# =========================================================
# EACH CONDITION
# =========================================================

for class_number in range(4):

    values = node_importance[
        class_number
    ]

    # -----------------------------------------------------
    # Sort by absolute importance
    # -----------------------------------------------------

    top_idx = np.argsort(
        np.abs(values)
    )[::-1][:TOP_N]


    print(
        "\n======================================"
    )

    print(
        CONDITION_NAMES[class_number]
    )

    print(
        "======================================"
    )

    print(
        f"{'Rank':<6}"
        f"{'Node':<8}"
        f"{'Label':<35}"
        f"{'Importance':>15}"
    )

    print(
        "-" * 70
    )


    for rank, idx in enumerate(
        top_idx,
        start=1
    ):

        print(
            f"{rank:<6}"
            f"{idx:<8}"
            f"{labels[idx]:<35}"
            f"{values[idx]:>15.6f}"
        )
# =========================================================
# ABSOLUTE NODE IMPORTANCE
# =========================================================

mean_abs_node_importance = np.mean(
    np.abs(
        node_importance
    ),
    axis=0
)

plt.figure(
    figsize=(14, 5)
)

plt.bar(
    np.arange(
        len(mean_abs_node_importance)
    ),
    mean_abs_node_importance
)

plt.xlabel(
    "MEG node"
)

plt.ylabel(
    "Mean |Gradient × Input|"
)

plt.title(
    "300–400 ms: Overall node importance"
)

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_overall_node_importance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# CLASSIFIER WEIGHTS
# =========================================================

print("\n======================================")
print("CLASSIFIER WEIGHTS")
print("======================================")

print(
    "Shape:",
    classifier_weights.shape
)

# Average across participants
mean_classifier_weights = np.mean(
    classifier_weights,
    axis=0
)

print(
    "Group mean shape:",
    mean_classifier_weights.shape
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    mean_classifier_weights,
    aspect="auto"
)

plt.colorbar(
    label="Weight"
)

plt.xlabel(
    "Output class"
)

plt.ylabel(
    "Dense unit"
)

plt.xticks(
    range(4),
    CONDITION_NAMES,
    rotation=45,
    ha="right"
)

plt.title(
    "300–400 ms: Group mean final classifier weights"
)

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_classifier_weights.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# LSTM INPUT KERNEL
# =========================================================

print("\n======================================")
print("LSTM INPUT KERNEL")
print("======================================")

print(
    "Shape:",
    lstm_kernel.shape
)

mean_lstm_kernel = np.mean(
    lstm_kernel,
    axis=0
)

print(
    "Group mean shape:",
    mean_lstm_kernel.shape
)


# ---------------------------------------------------------
# LSTM has 4 gates
#
# columns:
#
# 0:32      input gate
# 32:64     forget gate
# 64:96     cell/update gate
# 96:128    output gate
# ---------------------------------------------------------

gate_names = [
    "Input gate",
    "Forget gate",
    "Cell/update gate",
    "Output gate"
]

for gate_number, gate_name in enumerate(
    gate_names
):

    start = (
        gate_number * LSTM_UNITS
        if "LSTM_UNITS" in globals()
        else gate_number * 32
    )

    end = start + (
        LSTM_UNITS
        if "LSTM_UNITS" in globals()
        else 32
    )

    gate_weights = mean_lstm_kernel[
        :,
        start:end
    ]

    node_strength = np.mean(
        np.abs(
            gate_weights
        ),
        axis=1
    )

    plt.figure(
        figsize=(14, 5)
    )

    plt.bar(
        np.arange(
            len(node_strength)
        ),
        node_strength
    )

    plt.xlabel(
        "MEG node"
    )

    plt.ylabel(
        "Mean absolute weight"
    )

    plt.title(
        f"300–400 ms: LSTM {gate_name}"
    )

    plt.tight_layout()

    plt.savefig(
        MODEL_PATH /
        f"300_400ms_LSTM_{gate_name.replace('/', '_')}.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()


# =========================================================
# LSTM RECURRENT WEIGHTS
# =========================================================

print("\n======================================")
print("LSTM RECURRENT WEIGHTS")
print("======================================")

print(
    "Shape:",
    lstm_recurrent_kernel.shape
)

mean_recurrent = np.mean(
    lstm_recurrent_kernel,
    axis=0
)

plt.figure(
    figsize=(8, 8)
)

plt.imshow(
    mean_recurrent,
    aspect="auto"
)

plt.colorbar(
    label="Weight"
)

plt.xlabel(
    "LSTM unit / gate"
)

plt.ylabel(
    "LSTM unit"
)

plt.title(
    "300–400 ms: Group mean LSTM recurrent weights"
)

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_LSTM_recurrent_weights.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# SCALER
# =========================================================

print("\n======================================")
print("SCALER")
print("======================================")

print(
    "Scaler mean shape:",
    scaler_means.shape
)

print(
    "Scaler scale shape:",
    scaler_scales.shape
)

mean_scaler = np.mean(
    scaler_means,
    axis=0
)

mean_scale = np.mean(
    scaler_scales,
    axis=0
)

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    mean_scaler,
    label="Mean"
)

plt.plot(
    mean_scale,
    label="Scale"
)

plt.xlabel(
    "MEG node"
)

plt.ylabel(
    "Value"
)

plt.title(
    "300–400 ms: Group mean scaler parameters"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    MODEL_PATH /
    "300_400ms_scaler_parameters.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# SUMMARY
# =========================================================

print("\n======================================")
print("300–400 ms ANALYSIS COMPLETE")
print("======================================")

print(
    f"Mean accuracy = {mean_accuracy:.4f}"
)

print(
    f"SEM            = {sem_accuracy:.4f}"
)

print(
    f"Participants    = {len(participant_accuracy)}"
)

print(
    "\nAll plots saved in:"
)

print(
    MODEL_PATH
)