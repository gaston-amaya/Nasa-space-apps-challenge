#!/usr/bin/env python3
import io
import os
import re
import time
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

FEATURE_NAMES = [
    "hr_bpm",
    "spo2_pct",
    "bp_sys",
    "temp_c",
    "glucose_mg_dl",
    "rad_usv_h",
    "co2_ppm",
    "sleep_hrs",
    "exercise_mins",
    "mood_score",
]

CLASS_NAMES = [
    "0: NOMINAL (OK)",
    "1: CARDIO / HYPOXIA (+)",
    "2: RADIATION / ENVIRONMENT (/!\\)",
    "3: NEURO / FATIGUE (C)",
    "4: METABOLIC / BONE LOSS (<>)",
]

RADLAB_API_URL = (
    "https://visualization.osdr.nasa.gov/radlab/api/"
    "?spacecraft=ISS&instrument=REM&absorbed_dose_rate&format=csv"
)


def fetch_nasa_radlab_telemetry(num_samples: int = 2000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    print(f"[1/4] Fetching NASA OSDR RadLab ISS telemetry from:\n      {RADLAB_API_URL}")

    try:
        response = requests.get(RADLAB_API_URL, timeout=10)
        response.raise_for_status()
        text = response.text.strip()

        numeric_vals = []
        try:
            df_rad = pd.read_csv(io.StringIO(text))
            for col in df_rad.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ("dose", "rate", "val", "rad", "rem")):
                    vals = pd.to_numeric(df_rad[col], errors="coerce").dropna().to_numpy()
                    vals = vals[np.isfinite(vals) & (vals > 0)]
                    if len(vals) >= 10:
                        numeric_vals = vals
                        break
            if len(numeric_vals) < 10:
                for col in df_rad.select_dtypes(include=[np.number]).columns:
                    vals = df_rad[col].dropna().to_numpy()
                    vals = vals[np.isfinite(vals) & (vals > 0) & (vals < 1e4)]
                    if len(vals) >= 10:
                        numeric_vals = vals
                        break
        except Exception:
            pass

        if len(numeric_vals) < 10:
            floats = [
                float(m)
                for m in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", text)
            ]
            vals = np.array([v for v in floats if 0.0 < v < 1000.0], dtype=np.float64)
            if len(vals) >= 10:
                numeric_vals = vals

        if len(numeric_vals) >= 10:
            # Scale raw readings to nominal ISS dose rate (5.0 - 22.0 uSv/h) using 5th/95th percentiles
            vmin, vmax = np.percentile(numeric_vals, 5), np.percentile(numeric_vals, 95)
            if vmax - vmin < 1e-9:
                scaled = np.full_like(numeric_vals, 12.5)
            else:
                norm = np.clip((numeric_vals - vmin) / (vmax - vmin), 0.0, 1.0)
                scaled = 5.5 + norm * (21.5 - 5.5)
            sampled = rng.choice(scaled, size=num_samples, replace=True)
            sampled += rng.normal(0.0, 0.35, size=num_samples)
            sampled = np.clip(sampled, 5.0, 22.0)
            print(
                f"      -> Loaded {len(numeric_vals)} real ISS RadLab readings "
                f"(scaled mean: {sampled.mean():.2f} uSv/h)."
            )
            return sampled
        else:
            raise ValueError("Insufficient numeric dose rows in API response")

    except Exception as exc:
        print(f"      -> NASA RadLab API unreachable or timed out ({exc}).")
        print("      -> Using realistic ISS orbital radiation fallback generator (5.0 - 22.0 uSv/h).")
        orbit_phase = rng.uniform(0, 2 * np.pi, size=num_samples)
        base_dose = 11.5 + 5.5 * np.sin(orbit_phase) + 2.5 * np.cos(2 * orbit_phase)
        noise = rng.normal(0.0, 1.1, size=num_samples)
        return np.clip(base_dose + noise, 5.0, 22.0)


def make_nominal_block(n: int, rng: np.random.Generator, iss_rad_pool: np.ndarray) -> np.ndarray:
    X = np.zeros((n, 10), dtype=np.float64)
    X[:, 0] = np.clip(rng.normal(74.0, 6.5, n), 58.0, 92.0)
    X[:, 1] = np.clip(rng.normal(98.0, 0.9, n), 95.0, 100.0)
    X[:, 2] = np.clip(rng.normal(117.0, 5.0, n), 105.0, 128.0)
    X[:, 3] = np.clip(rng.normal(36.85, 0.18, n), 36.4, 37.3)
    X[:, 4] = np.clip(rng.normal(94.0, 8.0, n), 75.0, 115.0)
    X[:, 5] = np.clip(rng.choice(iss_rad_pool, size=n, replace=True) + rng.normal(0, 0.3, n), 5.0, 22.0)
    X[:, 6] = np.clip(rng.normal(1350.0, 420.0, n), 400.0, 2500.0)
    X[:, 7] = np.clip(rng.normal(7.5, 0.42, n), 6.5, 8.5)
    X[:, 8] = np.clip(rng.normal(88.0, 13.0, n), 60.0, 120.0)
    X[:, 9] = np.clip(np.round(rng.uniform(2.8, 5.0, n)) + rng.normal(0, 0.08, n), 3.0, 5.0)
    return X


def generate_dataset(iss_rad_pool: np.ndarray, seed: int = 42) -> pd.DataFrame:
    print("[2/4] Generating 10,000-row physiological telemetry dataset (astronaut_10stats.csv)...")
    rng = np.random.default_rng(seed)
    n_per_class = 2000

    # Class 0: Nominal
    X0 = make_nominal_block(n_per_class, rng, iss_rad_pool)
    y0 = np.zeros(n_per_class, dtype=int)

    # Class 1: Cardio / Hypoxia (SpO2 < 92%, HR > 110 or < 48, BP > 142 or < 90)
    X1 = make_nominal_block(n_per_class, rng, iss_rad_pool)
    X1[0:400, 1] = rng.uniform(80.0, 91.7, 400) + rng.normal(0, 0.2, 400)
    X1[400:750, 0] = rng.uniform(111.0, 160.0, 350) + rng.normal(0, 1.0, 350)
    X1[750:900, 0] = rng.uniform(36.0, 47.5, 150) + rng.normal(0, 0.5, 150)
    X1[900:1250, 2] = rng.uniform(143.0, 182.0, 350) + rng.normal(0, 1.0, 350)
    X1[1250:1400, 2] = rng.uniform(72.0, 89.0, 150) + rng.normal(0, 0.8, 150)
    X1[1400:2000, 1] = rng.uniform(82.0, 91.5, 600) + rng.normal(0, 0.3, 600)
    X1[1400:2000, 0] = rng.uniform(112.0, 152.0, 600) + rng.normal(0, 1.5, 600)
    X1[1400:2000, 2] = rng.uniform(138.0, 172.0, 600) + rng.normal(0, 1.5, 600)
    y1 = np.full(n_per_class, 1, dtype=int)

    # Class 2: Radiation / Environment (Rad > 50, CO2 > 4000, Temp > 37.8)
    X2 = make_nominal_block(n_per_class, rng, iss_rad_pool)
    X2[0:450, 5] = rng.uniform(51.0, 185.0, 450) + rng.normal(0, 1.5, 450)
    X2[450:900, 6] = rng.uniform(4050.0, 8200.0, 450) + rng.normal(0, 40.0, 450)
    X2[900:1300, 3] = rng.uniform(37.85, 39.7, 400) + rng.normal(0, 0.05, 400)
    X2[1300:1700, 3] = rng.uniform(37.82, 39.4, 400) + rng.normal(0, 0.05, 400)
    X2[1300:1700, 5] = rng.uniform(42.0, 150.0, 400) + rng.normal(0, 1.5, 400)
    X2[1300:1700, 0] = rng.uniform(92.0, 108.0, 400) + rng.normal(0, 1.0, 400)
    X2[1700:2000, 5] = rng.uniform(55.0, 175.0, 300) + rng.normal(0, 1.5, 300)
    X2[1700:2000, 6] = rng.uniform(3800.0, 7200.0, 300) + rng.normal(0, 35.0, 300)
    X2[1700:2000, 3] = rng.uniform(37.5, 38.8, 300) + rng.normal(0, 0.05, 300)
    y2 = np.full(n_per_class, 2, dtype=int)

    # Class 3: Neuro / Fatigue (Sleep < 5.0, Mood 1-2)
    X3 = make_nominal_block(n_per_class, rng, iss_rad_pool)
    X3[0:550, 7] = rng.uniform(2.0, 4.9, 550) + rng.normal(0, 0.08, 550)
    X3[550:1100, 9] = np.clip(np.round(rng.uniform(1.0, 2.0, 550)) + rng.normal(0, 0.05, 550), 1.0, 2.1)
    X3[1100:2000, 7] = rng.uniform(2.2, 4.9, 900) + rng.normal(0, 0.08, 900)
    X3[1100:2000, 9] = np.clip(np.round(rng.uniform(1.0, 2.0, 900)) + rng.normal(0, 0.05, 900), 1.0, 2.1)
    X3[1100:2000, 0] = rng.uniform(88.0, 106.0, 900) + rng.normal(0, 1.0, 900)
    y3 = np.full(n_per_class, 3, dtype=int)

    # Class 4: Metabolic / Bone Loss (Glucose < 70 or > 145, Exercise < 30)
    X4 = make_nominal_block(n_per_class, rng, iss_rad_pool)
    X4[0:550, 4] = rng.uniform(42.0, 69.2, 550) + rng.normal(0, 0.5, 550)
    X4[550:900, 4] = rng.uniform(146.0, 215.0, 350) + rng.normal(0, 1.0, 350)
    X4[900:1450, 8] = np.clip(rng.uniform(0.0, 29.0, 550) + rng.normal(0, 0.5, 550), 0.0, 29.5)
    X4[1450:1800, 4] = rng.uniform(45.0, 68.5, 350) + rng.normal(0, 0.5, 350)
    X4[1450:1800, 8] = np.clip(rng.uniform(5.0, 28.5, 350) + rng.normal(0, 0.5, 350), 0.0, 29.5)
    X4[1800:2000, 4] = rng.uniform(147.0, 205.0, 200) + rng.normal(0, 1.0, 200)
    X4[1800:2000, 8] = np.clip(rng.uniform(5.0, 28.5, 200) + rng.normal(0, 0.5, 200), 0.0, 29.5)
    y4 = np.full(n_per_class, 4, dtype=int)

    X = np.vstack([X0, X1, X2, X3, X4])
    y = np.concatenate([y0, y1, y2, y3, y4])

    X[:, 0] = np.round(X[:, 0], 1)
    X[:, 1] = np.round(np.clip(X[:, 1], 70.0, 100.0), 1)
    X[:, 2] = np.round(X[:, 2], 1)
    X[:, 3] = np.round(X[:, 3], 2)
    X[:, 4] = np.round(X[:, 4], 1)
    X[:, 5] = np.round(np.clip(X[:, 5], 1.0, 300.0), 2)
    X[:, 6] = np.round(np.clip(X[:, 6], 300.0, 10000.0), 0)
    X[:, 7] = np.round(np.clip(X[:, 7], 0.5, 12.0), 2)
    X[:, 8] = np.round(np.clip(X[:, 8], 0.0, 180.0), 1)
    X[:, 9] = np.round(np.clip(X[:, 9], 1.0, 5.0), 1)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["alert_class"] = y

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    csv_path = "astronaut_10stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"      -> Saved {len(df):,} shuffled rows to '{csv_path}'.")
    return df


def train_neural_network(df: pd.DataFrame):
    print("[3/4] Training MLP Neural Network (10 -> 32 -> 16 -> 5) on local CPU...")
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["alert_class"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=150,
        random_state=42,
    )
    clf.fit(X_train_scaled, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"      -> CPU Training Time : {elapsed:.3f} seconds")
    print(f"      -> Test Set Accuracy : {acc * 100:.2f}%\n")
    print("Classification Report (2,000 held-out test samples):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
    return scaler, clf


def format_1d_array(name: str, arr: np.ndarray) -> str:
    vals = ", ".join(f"{float(v):.7f}f" for v in arr)
    return f"const float {name}[{len(arr)}] = {{ {vals} }};\n"


def format_2d_array(name: str, arr: np.ndarray) -> str:
    rows, cols = arr.shape
    lines = [f"const float {name}[{rows}][{cols}] = {{"]
    for r in range(rows):
        row_str = ", ".join(f"{float(v):.7f}f" for v in arr[r])
        comma = "," if r < rows - 1 else ""
        lines.append(f"    {{ {row_str} }}{comma}")
    lines.append("};\n")
    return "\n".join(lines)


def export_cpp_header(scaler: StandardScaler, clf: MLPClassifier, out_dir: str = "./esp32_triage"):
    os.makedirs(out_dir, exist_ok=True)
    header_path = os.path.join(out_dir, "model_weights.h")
    print(f"[4/4] Exporting zero-dependency C++ header to '{header_path}'...")

    W1, W2, W3 = clf.coefs_[0], clf.coefs_[1], clf.coefs_[2]
    B1, B2, B3 = clf.intercepts_[0], clf.intercepts_[1], clf.intercepts_[2]
    mean = scaler.mean_
    std = scaler.scale_

    header_content = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <math.h>

{format_1d_array("FEATURE_MEAN", mean)}
{format_1d_array("FEATURE_STD", std)}
{format_2d_array("MLP_W1", W1)}
{format_1d_array("MLP_B1", B1)}
{format_2d_array("MLP_W2", W2)}
{format_1d_array("MLP_B2", B2)}
{format_2d_array("MLP_W3", W3)}
{format_1d_array("MLP_B3", B3)}
// Forward pass: Z-score normalization -> 32 ReLU -> 16 ReLU -> 5-class Softmax
inline int run_astronaut_triage(const float input[10], float* confidence_out, float probs_out[5] = nullptr) {{
    float x[10];
    for (int i = 0; i < 10; ++i) {{
        x[i] = (input[i] - FEATURE_MEAN[i]) / FEATURE_STD[i];
    }}

    float h1[32];
    for (int j = 0; j < 32; ++j) {{
        float acc = MLP_B1[j];
        for (int i = 0; i < 10; ++i) {{
            acc += x[i] * MLP_W1[i][j];
        }}
        h1[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float h2[16];
    for (int j = 0; j < 16; ++j) {{
        float acc = MLP_B2[j];
        for (int i = 0; i < 32; ++i) {{
            acc += h1[i] * MLP_W2[i][j];
        }}
        h2[j] = (acc > 0.0f) ? acc : 0.0f;
    }}

    float logits[5];
    float max_logit = -1e30f;
    for (int k = 0; k < 5; ++k) {{
        float acc = MLP_B3[k];
        for (int i = 0; i < 16; ++i) {{
            acc += h2[i] * MLP_W3[i][k];
        }}
        logits[k] = acc;
        if (acc > max_logit) {{
            max_logit = acc;
        }}
    }}

    float sum_exp = 0.0f;
    float probs[5];
    for (int k = 0; k < 5; ++k) {{
        probs[k] = expf(logits[k] - max_logit);
        sum_exp += probs[k];
    }}

    int best_class = 0;
    float best_prob = 0.0f;
    for (int k = 0; k < 5; ++k) {{
        probs[k] /= sum_exp;
        if (probs_out != nullptr) {{
            probs_out[k] = probs[k];
        }}
        if (probs[k] > best_prob) {{
            best_prob = probs[k];
            best_class = k;
        }}
    }}

    if (confidence_out != nullptr) {{
        *confidence_out = best_prob;
    }}
    return best_class;
}}

#endif
"""

    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header_content)
    print(f"      -> Successfully wrote '{header_path}' ({os.path.getsize(header_path):,} bytes).")


def verify_esp32_scenarios(scaler: StandardScaler, clf: MLPClassifier):
    print("\n[VERIFY] Testing ESP32 Presets & Single-Key Serial Monitor Overrides:")
    nominal = [72.0, 98.2, 116.0, 36.8, 92.0, 11.4, 1180.0, 7.6, 85.0, 4.0]
    test_cases = [
        ("Preset 0: Nominal",             nominal, 0),
        ("Preset 1: EVA Hypoxia",         [132.0, 88.5, 149.0, 36.9, 95.0, 12.0, 1350.0, 7.2, 78.0, 4.0], 1),
        ("Preset 2: Solar Rad Storm",     [98.0, 97.4, 122.0, 38.3, 96.0, 112.0, 4450.0, 7.0, 75.0, 3.0], 2),
        ("Preset 3: Circadian Exhaustion",[99.0, 97.8, 121.0, 36.9, 90.0, 13.2, 1420.0, 3.8, 68.0, 1.0], 3),
        ("Preset 4: Acute Hypoglycemia",  [78.0, 98.0, 114.0, 36.7, 58.0, 11.8, 1250.0, 7.4, 16.0, 3.0], 4),
    ]
    overrides = [
        ("Serial Cmd HR:128",    0, 128.0,  1),
        ("Serial Cmd SPO2:89",   1, 89.0,   1),
        ("Serial Cmd BP:148",    2, 148.0,  1),
        ("Serial Cmd TEMP:38.4", 3, 38.4,   2),
        ("Serial Cmd GLU:62",    4, 62.0,   4),
        ("Serial Cmd GLU:60",    4, 60.0,   4),
        ("Serial Cmd RAD:95",    5, 95.0,   2),
        ("Serial Cmd RAD:120",   5, 120.0,  2),
        ("Serial Cmd CO2:4500",  6, 4500.0, 2),
        ("Serial Cmd SLEEP:4.0", 7, 4.0,    3),
        ("Serial Cmd EX:15",     8, 15.0,   4),
        ("Serial Cmd MOOD:1",    9, 1.0,    3),
    ]
    for label, idx, val, exp_cls in overrides:
        vec = list(nominal)
        vec[idx] = val
        test_cases.append((label, vec, exp_cls))

    X_cases = np.array([tc[1] for tc in test_cases], dtype=np.float64)
    probs = clf.predict_proba(scaler.transform(X_cases))
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    for (label, _, exp_cls), pred_cls, conf in zip(test_cases, preds, confs):
        status = "PASS" if pred_cls == exp_cls else "FAIL"
        print(f"      [{status}] {label:32s} -> Class {pred_cls} (Expected {exp_cls}) | Conf: {conf * 100:5.1f}%")


if __name__ == "__main__":
    iss_rad = fetch_nasa_radlab_telemetry(num_samples=2000, seed=42)
    dataset = generate_dataset(iss_rad, seed=42)
    scaler_model, mlp_model = train_neural_network(dataset)
    export_cpp_header(scaler_model, mlp_model, out_dir="./esp32_triage")
    verify_esp32_scenarios(scaler_model, mlp_model)
