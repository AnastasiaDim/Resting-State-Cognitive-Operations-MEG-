#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-class LSTM MEG decoding with temporal-window comparison

Conditions
----------
9  = Stay Left
13 = Stay Right
17 = Shift Left
21 = Shift Right

Temporal windows
----------------
0-100 ms
100-200 ms
...
900-1000 ms

For every temporal window the script calculates:

1. Participant accuracy
2. Participant training/validation curves
3. Participant confusion matrices
4. Group mean accuracy
5. Group confusion matrix
6. Group normalized confusion matrix
7. Condition-specific Gradient × Input node importance
8. Group node importance
9. Actual trained LSTM model
10. Scaler used for that model
11. LSTM weights
12. Final classifier weights

The saved models/scalers can later be applied to
resting-state MEG data.
"""

# =========================================================
# IMPORTS
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mne
import tensorflow as tf

from nilearn import plotting

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = Path(
    "/home/uranus/Scrivania/MEG_replay/epoched_data"
)

# Folder containing trained models and weights
MODEL_PATH = DATA_PATH / "lstm_temporal_models"

MODEL_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# CONDITIONS
# =========================================================

CONDITIONS = {
    "stay_left": "9",
    "stay_right": "13",
    "shift_left": "17",
    "shift_right": "21"
}

CONDITION_NAMES = [
    "Stay Left",
    "Stay Right",
    "Shift Left",
    "Shift Right"
]

N_CLASSES = 4




WINDOWS = [
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0)
]


# =========================================================
# CLASSIFICATION SETTINGS
# =========================================================

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42

MAX_EPOCHS = 100
BATCH_SIZE = 16

LSTM_UNITS = 32
DENSE_UNITS = 16

DROPOUT = 0.30

PATIENCE = 10


# =========================================================
# REPRODUCIBILITY
# =========================================================

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# =========================================================
# HELPER
# =========================================================

def pad_histories(histories):

    max_length = max(
        len(h)
        for h in histories
    )

    padded = np.full(
        (
            len(histories),
            max_length
        ),
        np.nan
    )

    for i, h in enumerate(histories):

        padded[
            i,
            :len(h)
        ] = h

    return padded


# =========================================================
# HELPER: WINDOW NAME
# =========================================================

def window_name(
    start,
    end
):

    start_ms = int(
        round(start * 1000)
    )

    end_ms = int(
        round(end * 1000)
    )

    return (
        f"{start_ms:04d}_{end_ms:04d}ms"
    )


# =========================================================
# LOAD FILES
# =========================================================

files = sorted(
    DATA_PATH.glob(
        "Sub*_epo.fif"
    )
)

if len(files) == 0:

    raise FileNotFoundError(
        f"No files matching Sub*_epo.fif found in:\n"
        f"{DATA_PATH}"
    )


print(
    "\n======================================"
)

print(
    "TEMPORAL-WINDOW LSTM DECODING"
)

print(
    "======================================"
)

print(
    f"Participants found: {len(files)}"
)

print(
    "\nWindows:"
)

for start, end in WINDOWS:

    print(
        f"  {start*1000:.0f}–{end*1000:.0f} ms"
    )


# =========================================================
# GLOBAL STORAGE
# =========================================================

# Results for every window
window_results = {}

# =========================================================
# MAIN WINDOW LOOP
# =========================================================

for window_number, (
    WINDOW_START,
    WINDOW_END
) in enumerate(
    WINDOWS,
    start=1
):

    WIN_NAME = window_name(
        WINDOW_START,
        WINDOW_END
    )

    print(
        "\n\n"
        "################################################"
    )

    print(
        f"WINDOW {window_number}/{len(WINDOWS)}"
    )

    print(
        f"{WINDOW_START*1000:.0f}–"
        f"{WINDOW_END*1000:.0f} ms"
    )

    print(
        "################################################"
    )


    # =====================================================
    # STORAGE FOR THIS WINDOW
    # =====================================================

    participant_acc = []

    all_confusion_matrices = []

    all_train_histories = []

    all_val_histories = []

    all_node_importance = []

    all_classifier_weights = []

    all_classifier_bias = []

    all_lstm_kernels = []

    all_lstm_recurrent_kernels = []

    all_lstm_biases = []

    all_scaler_means = []

    all_scaler_scales = []

    participant_names = []


    # =====================================================
    # PARTICIPANT LOOP
    # =====================================================

    for participant_number, f in enumerate(
        files,
        start=1
    ):

        print(
            "\n--------------------------------"
        )

        print(
            f"{f.stem}"
        )

        print(
            f"Window: "
            f"{WINDOW_START*1000:.0f}–"
            f"{WINDOW_END*1000:.0f} ms"
        )

        print(
            "--------------------------------"
        )


        # =================================================
        # LOAD EPOCHS
        # =================================================

        epochs = mne.read_epochs(
            f,
            preload=True,
            verbose=False
        )


        # =================================================
        # MAKE SURE DATA COVER 0–1 SECOND
        # =================================================

        if epochs.times[-1] < 1.0:

            print(
                "WARNING: Epoch does not reach 1 second."
            )

            print(
                f"Available time: "
                f"{epochs.times[0]:.3f}–"
                f"{epochs.times[-1]:.3f}"
            )

            continue


        # =================================================
        # CHECK SAMPLING RATE
        # =================================================

        sfreq = epochs.info[
            "sfreq"
        ]

        print(
            f"Sampling frequency: {sfreq:.1f} Hz"
        )


        # =================================================
        # LOAD FOUR CONDITIONS
        # =================================================

        condition_data = {}

        valid_participant = True

        for condition, event_code in CONDITIONS.items():

            try:

                data = epochs[
                    event_code
                ].get_data()

            except Exception as e:

                print(
                    f"Could not load {condition}: {e}"
                )

                valid_participant = False

                break

            condition_data[
                condition
            ] = data

            print(
                f"{condition:12s}: "
                f"{data.shape[0]} trials"
            )

        if not valid_participant:

            continue


        # =================================================
        # BALANCE CLASSES
        # =================================================

        n_min = min(
            len(
                condition_data[c]
            )
            for c in CONDITIONS
        )

        print(
            f"Trials per class: {n_min}"
        )

        if n_min < 5:

            print(
                "Too few trials. Skipping."
            )

            continue


        # =================================================
        # RANDOM BALANCING
        # =================================================

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        balanced_data = []

        labels = []

        for class_number, condition in enumerate(
            CONDITIONS
        ):

            data = condition_data[
                condition
            ]

            idx = rng.choice(
                len(data),
                size=n_min,
                replace=False
            )

            data = data[
                idx
            ]

            balanced_data.append(
                data
            )

            labels.extend(
                [class_number] * len(data)
            )


        # =================================================
        # COMBINE
        # =================================================

        X_full = np.vstack(
            balanced_data
        )

        y = np.array(
            labels
        )

        print(
            "Full data shape:",
            X_full.shape
        )


        # =================================================
        # SHUFFLE
        # =================================================

        shuffle_idx = rng.permutation(
            len(X_full)
        )

        X_full = X_full[
            shuffle_idx
        ]

        y = y[
            shuffle_idx
        ]


        # =================================================
        # EXTRACT THIS TEMPORAL WINDOW
        # =================================================
        #
        # Original:
        #
        # trials × nodes × time
        #
        # Crop the selected time period.
        #
        # =================================================

        # Convert times to sample indices.
        #
        # This avoids MNE's inclusive tmax issue.

        start_sample = int(
            round(
                WINDOW_START * sfreq
            )
        )

        end_sample = int(
            round(
                WINDOW_END * sfreq
            )
        )

        X_full = X_full[
            :,
            :,
            start_sample:end_sample
        ]


        # =================================================
        # TRANSPOSE FOR LSTM
        # =================================================
        #
        # trials × time × nodes
        # =================================================

        X = np.transpose(
            X_full,
            (0, 2, 1)
        )

        n_trials, n_times, n_features = X.shape

        print(
            "LSTM input shape:",
            X.shape
        )


        # =================================================
        # TRAIN / TEST SPLIT
        # =================================================

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                stratify=y,
                random_state=RANDOM_STATE
            )
        )


        # =================================================
        # NORMALIZATION
        # =================================================
        #
        # IMPORTANT:
        #
        # Fit ONLY on training data.
        #
        #
        # =================================================

        X_train_2d = X_train.reshape(
            -1,
            n_features
        )

        X_test_2d = X_test.reshape(
            -1,
            n_features
        )

        scaler = StandardScaler()

        X_train_2d = scaler.fit_transform(
            X_train_2d
        )

        X_test_2d = scaler.transform(
            X_test_2d
        )

        X_train = X_train_2d.reshape(
            X_train.shape
        )

        X_test = X_test_2d.reshape(
            X_test.shape
        )


        # =================================================
        # SAVE SCALER
        # =================================================

        scaler_path = MODEL_PATH / (
            f"{f.stem}_{WIN_NAME}_scaler.npz"
        )

        np.savez(
            scaler_path,

            mean=scaler.mean_,

            scale=scaler.scale_
        )


        # =================================================
        # CREATE LSTM
        # =================================================

        model = Sequential()

        model.add(
            LSTM(
                LSTM_UNITS,

                input_shape=(
                    n_times,
                    n_features
                ),

                dropout=DROPOUT,

                recurrent_dropout=DROPOUT
            )
        )

        model.add(
            Dense(
                DENSE_UNITS,
                activation="relu"
            )
        )

        model.add(
            Dropout(
                DROPOUT
            )
        )

        model.add(
            Dense(
                N_CLASSES,
                activation="softmax"
            )
        )


        # =================================================
        # COMPILE
        # =================================================

        model.compile(
            optimizer="adam",

            loss=(
                "sparse_categorical_crossentropy"
            ),

            metrics=[
                "accuracy"
            ]
        )


        # =================================================
        # EARLY STOPPING
        # =================================================

        early_stop = EarlyStopping(
            monitor="val_loss",

            patience=PATIENCE,

            restore_best_weights=True
        )


        # =================================================
        # TRAIN
        # =================================================

        history = model.fit(

            X_train,

            y_train,

            validation_split=VALIDATION_SIZE,

            epochs=MAX_EPOCHS,

            batch_size=BATCH_SIZE,

            callbacks=[
                early_stop
            ],

            verbose=0
        )


        # =================================================
        # SAVE MODEL
        # =================================================

        model_path = MODEL_PATH / (
            f"{f.stem}_{WIN_NAME}_LSTM.keras"
        )

        model.save(
            model_path
        )


        # =================================================
        # STORE LEARNING CURVES
        # =================================================

        all_train_histories.append(
            history.history[
                "accuracy"
            ]
        )

        all_val_histories.append(
            history.history[
                "val_accuracy"
            ]
        )


        # =================================================
        # TEST
        # =================================================

        y_prob = model.predict(
            X_test,
            verbose=0
        )

        y_pred = np.argmax(
            y_prob,
            axis=1
        )

        acc = accuracy_score(
            y_test,
            y_pred
        )

        participant_acc.append(
            acc
        )

        participant_names.append(
            f.stem
        )

        print(
            f"Test accuracy: {acc:.3f}"
        )


        # =================================================
        # CONFUSION MATRIX
        # =================================================

        cm = confusion_matrix(
            y_test,

            y_pred,

            labels=np.arange(
                N_CLASSES
            )
        )

        all_confusion_matrices.append(
            cm
        )


        # =================================================
        # SAVE ACTUAL LSTM WEIGHTS
        # =================================================

        lstm_layer = model.layers[0]

        lstm_weights = (
            lstm_layer.get_weights()
        )

        all_lstm_kernels.append(
            lstm_weights[0]
        )

        all_lstm_recurrent_kernels.append(
            lstm_weights[1]
        )

        all_lstm_biases.append(
            lstm_weights[2]
        )


        # =================================================
        # SAVE FINAL CLASSIFIER WEIGHTS
        # =================================================

        classifier_layer = model.layers[-1]

        classifier_weights = (
            classifier_layer.get_weights()
        )

        classifier_W = (
            classifier_weights[0]
        )

        classifier_b = (
            classifier_weights[1]
        )

        all_classifier_weights.append(
            classifier_W
        )

        all_classifier_bias.append(
            classifier_b
        )


        # =================================================
        # SAVE PARTICIPANT WEIGHTS
        # =================================================

        weights_path = MODEL_PATH / (
            f"{f.stem}_{WIN_NAME}_weights.npz"
        )

        np.savez(
            weights_path,

            lstm_kernel=lstm_weights[0],

            lstm_recurrent_kernel=lstm_weights[1],

            lstm_bias=lstm_weights[2],

            classifier_weights=classifier_W,

            classifier_bias=classifier_b,

            scaler_mean=scaler.mean_,

            scaler_scale=scaler.scale_,

            condition_names=np.array(
                CONDITION_NAMES
            )
        )


        # =================================================
        # NODE IMPORTANCE
        # =================================================

        node_importance = np.full(
            (
                N_CLASSES,
                n_features
            ),
            np.nan
        )


        for class_number in range(
            N_CLASSES
        ):

            idx = (
                y_test == class_number
            )

            X_class = X_test[
                idx
            ]

            if len(X_class) == 0:

                continue


            X_tensor = tf.convert_to_tensor(
                X_class,
                dtype=tf.float32
            )


            with tf.GradientTape() as tape:

                tape.watch(
                    X_tensor
                )

                predictions = model(
                    X_tensor,
                    training=False
                )

                class_score = tf.reduce_sum(
                    predictions[
                        :,
                        class_number
                    ]
                )


            gradients = tape.gradient(
                class_score,
                X_tensor
            )


            if gradients is None:

                raise RuntimeError(
                    "Could not calculate gradients."
                )


            importance = (
                gradients.numpy()
                *
                X_class
            )


            node_importance[
                class_number
            ] = np.mean(
                importance,
                axis=(0, 1)
            )


        all_node_importance.append(
            node_importance
        )


        # =================================================
        # SAVE SCALER ARRAYS
        # =================================================

        all_scaler_means.append(
            scaler.mean_
        )

        all_scaler_scales.append(
            scaler.scale_
        )


    # =====================================================
    # CHECK WINDOW RESULTS
    # =====================================================

    if len(participant_acc) == 0:

        print(
            "\nNo participants for this window."
        )

        continue


    # =====================================================
    # CONVERT TO ARRAYS
    # =====================================================

    participant_acc = np.array(
        participant_acc
    )

    all_confusion_matrices = np.array(
        all_confusion_matrices
    )

    all_node_importance = np.array(
        all_node_importance
    )

    all_classifier_weights = np.array(
        all_classifier_weights
    )

    all_classifier_bias = np.array(
        all_classifier_bias
    )

    all_lstm_kernels = np.array(
        all_lstm_kernels
    )

    all_lstm_recurrent_kernels = np.array(
        all_lstm_recurrent_kernels
    )

    all_lstm_biases = np.array(
        all_lstm_biases
    )

    all_scaler_means = np.array(
        all_scaler_means
    )

    all_scaler_scales = np.array(
        all_scaler_scales
    )


    # =====================================================
    # GROUP CONFUSION MATRIX
    # =====================================================

    group_cm = np.sum(
        all_confusion_matrices,
        axis=0
    )


    # =====================================================
    # NORMALIZED CONFUSION MATRIX
    # =====================================================

    row_sums = group_cm.sum(
        axis=1,
        keepdims=True
    )

    group_cm_normalized = np.divide(
        group_cm,

        row_sums,

        out=np.zeros_like(
            group_cm,
            dtype=float
        ),

        where=row_sums != 0
    )


    # =====================================================
    # GROUP NODE IMPORTANCE
    # =====================================================

    group_node_importance = np.nanmean(
        all_node_importance,
        axis=0
    )


    # =====================================================
    # GROUP ACCURACY
    # =====================================================

    mean_accuracy = (
        participant_acc.mean()
    )

    sem_accuracy = (
        participant_acc.std(
            ddof=1
        )
        /
        np.sqrt(
            len(participant_acc)
        )
    )


    print(
        "\n================================"
    )

    print(
        f"RESULTS: {WIN_NAME}"
    )

    print(
        "================================"
    )

    print(
        f"N participants: "
        f"{len(participant_acc)}"
    )

    print(
        f"Mean accuracy: "
        f"{mean_accuracy:.3f}"
    )

    print(
        f"SEM: "
        f"{sem_accuracy:.3f}"
    )

    print(
        "Chance: 0.250"
    )


    # =====================================================
    # SAVE WINDOW RESULTS
    # =====================================================

    window_output = MODEL_PATH / (
        f"{WIN_NAME}_results.npz"
    )

    np.savez(

        window_output,

        participant_accuracy=(
            participant_acc
        ),

        participant_names=np.array(
            participant_names
        ),

        all_confusion_matrices=(
            all_confusion_matrices
        ),

        group_confusion_matrix=(
            group_cm
        ),

        group_confusion_matrix_normalized=(
            group_cm_normalized
        ),

        all_node_importance=(
            all_node_importance
        ),

        group_node_importance=(
            group_node_importance
        ),

        classifier_weights=(
            all_classifier_weights
        ),

        classifier_bias=(
            all_classifier_bias
        ),

        lstm_kernel=(
            all_lstm_kernels
        ),

        lstm_recurrent_kernel=(
            all_lstm_recurrent_kernels
        ),

        lstm_bias=(
            all_lstm_biases
        ),

        scaler_means=(
            all_scaler_means
        ),

        scaler_scales=(
            all_scaler_scales
        ),

        window_start=WINDOW_START,

        window_end=WINDOW_END,

        condition_names=np.array(
            CONDITION_NAMES
        )
    )


    # =====================================================
    # SAVE RESULT IN MEMORY FOR FINAL COMPARISON
    # =====================================================

    window_results[
        WIN_NAME
    ] = {

        "start": WINDOW_START,

        "end": WINDOW_END,

        "participant_accuracy": (
            participant_acc
        ),

        "mean_accuracy": (
            mean_accuracy
        ),

        "sem_accuracy": (
            sem_accuracy
        ),

        "confusion_matrix": (
            group_cm
        ),

        "normalized_confusion_matrix": (
            group_cm_normalized
        ),

        "node_importance": (
            group_node_importance
        )
    }


# =========================================================
# FINAL WINDOW COMPARISON
# =========================================================

if len(window_results) == 0:

    raise RuntimeError(
        "No temporal windows were successfully analysed."
    )


window_labels = list(
    window_results.keys()
)

window_means = np.array([
    window_results[w][
        "mean_accuracy"
    ]
    for w in window_labels
])

window_sems = np.array([
    window_results[w][
        "sem_accuracy"
    ]
    for w in window_labels
])


# =========================================================
# FIND BEST WINDOW
# =========================================================

best_idx = np.argmax(
    window_means
)

best_window = window_labels[
    best_idx
]

best_accuracy = window_means[
    best_idx
]


print(
    "\n\n"
    "=========================================="
)

print(
    "TEMPORAL WINDOW COMPARISON"
)

print(
    "=========================================="
)

for label, mean, sem in zip(
    window_labels,
    window_means,
    window_sems
):

    print(
        f"{label:15s} "
        f"accuracy = {mean:.3f} "
        f"± {sem:.3f}"
    )


print(
    "\n------------------------------------------"
)

print(
    f"BEST WINDOW: {best_window}"
)

print(
    f"Mean accuracy: {best_accuracy:.3f}"
)

print(
    "------------------------------------------"
)


# =========================================================
# TEMPORAL DECODING PLOT
# =========================================================

x = np.arange(
    len(window_labels)
)

plt.figure(
    figsize=(13, 6)
)

plt.errorbar(

    x,

    window_means,

    yerr=window_sems,

    marker="o",

    linewidth=2,

    capsize=4
)

plt.axhline(
    0.25,

    linestyle="--",

    linewidth=2,

    label="Chance (25%)"
)

plt.xticks(
    x,
    window_labels,
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


temporal_plot_path = MODEL_PATH / (
    "temporal_window_decoding_accuracy.png"
)

plt.savefig(
    temporal_plot_path,
    dpi=600,
    bbox_inches="tight"
)

plt.show()


# =========================================================
# SAVE TEMPORAL COMPARISON
# =========================================================

comparison_file = MODEL_PATH / (
    "temporal_window_comparison.npz"
)

np.savez(

    comparison_file,

    window_labels=np.array(
        window_labels
    ),

    window_starts=np.array([
        window_results[w][
            "start"
        ]
        for w in window_labels
    ]),

    window_ends=np.array([
        window_results[w][
            "end"
        ]
        for w in window_labels
    ]),

    mean_accuracy=window_means,

    sem_accuracy=window_sems,

    chance=0.25,

    best_window=best_window,

    best_accuracy=best_accuracy
)


# =========================================================
# FINAL SUMMARY TABLE
# =========================================================

summary_df = pd.DataFrame({

    "window": window_labels,

    "start_ms": [
        window_results[w]["start"] * 1000
        for w in window_labels
    ],

    "end_ms": [
        window_results[w]["end"] * 1000
        for w in window_labels
    ],

    "mean_accuracy": window_means,

    "SEM": window_sems,

    "above_chance": (
        window_means > 0.25
    )
})

print(
    "\n"
    "=========================================="
)

print(
    "SUMMARY"
)

print(
    "=========================================="
)

print(
    summary_df.to_string(
        index=False
    )
)


summary_csv = MODEL_PATH / (
    "temporal_window_summary.csv"
)

summary_df.to_csv(
    summary_csv,
    index=False
)


print(
    "\nSaved:"
)

print(
    temporal_plot_path
)

print(
    comparison_file
)

print(
    summary_csv
)

print(
    "\nAll participant models, scalers and weights "
    "were saved in:"
)

print(
    MODEL_PATH
)

print(
    "\nAnalysis finished."
)


from pathlib import Path
import numpy as np

MODEL_PATH = Path(
    "/home/uranus/Scrivania/MEG_replay/epoched_data/lstm_temporal_models"
)

print("\nFILES FOUND\n")

files = sorted(MODEL_PATH.glob("*"))

for f in files:
    print(f.name)

print("\n==============================")
print("RESULT FILES")
print("==============================")

result_files = sorted(
    MODEL_PATH.glob("*_results.npz")
)

for f in result_files:
    print(f.name)

print(
    f"\nNumber of result files: {len(result_files)}"
)

print("\n==============================")
print("WEIGHT FILES")
print("==============================")

weight_files = sorted(
    MODEL_PATH.glob("*_weights.npz")
)

print(
    f"Number of weight files: {len(weight_files)}"
)

for f in weight_files:
    print(f.name)