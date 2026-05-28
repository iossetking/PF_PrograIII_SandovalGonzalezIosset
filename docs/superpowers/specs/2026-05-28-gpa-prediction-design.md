# GPA Post-AI Prediction — Design Spec

**Date:** 2026-05-28
**Project:** PF_PrograIII_SandovalGonzalezIosset
**Dataset:** AI_Impact_Student_Life_2026.csv (1,500 rows, 11 columns)

---

## Goal

Train a neural network to predict `GPA_Post_AI` (a student's GPA after using AI tools) from student profile and AI usage features. Output MAE/RMSE metrics on a held-out test set and predict GPA for a hardcoded sample student.

---

## Features

### Input features (9 total)
| Feature | Type | Preprocessing |
|---|---|---|
| `Age` | int | MinMaxScaler |
| `Major` | category (6) | One-Hot Encoding |
| `Primary_AI_Tool` | category (5) | One-Hot Encoding |
| `Task_Frequency_Daily` | int | MinMaxScaler |
| `Main_Usage_Case` | category (5) | One-Hot Encoding |
| `GPA_Baseline` | float | MinMaxScaler |
| `Time_Saved_Hours_Weekly` | int | MinMaxScaler |
| `AI_Ethics_Concern` | category (3) | One-Hot Encoding |
| `Career_Confidence_Score` | int | MinMaxScaler |

`Student_ID` is dropped. `GPA_Post_AI` is the target (not a feature).

### Target
`GPA_Post_AI` — float, range 2.4–4.0.

---

## Architecture

### File layout (no new files)
```
data.py   — load, preprocess, split
model.py  — build, train
main.py   — orchestrate, evaluate, predict sample
```

### Preprocessing — data.py
- Load `AI_Impact_Student_Life_2026.csv` with pandas
- Drop `Student_ID`
- One-Hot Encode 4 categorical columns via `pd.get_dummies()`
- `MinMaxScaler` on 5 numeric columns
- 80/20 train-test split (`random_state=42`)
- Return `X_train, X_test, y_train, y_test, scaler, ohe_columns`
  - `scaler` and `ohe_columns` are returned so `main.py` can preprocess a sample student identically

### Network — model.py
```
Input(shape=[n_features])
Dense(64, activation='relu')
Dense(32, activation='relu')
Dense(1)                      # linear output — regression
```
- Loss: `mean_squared_error`
- Optimizer: `Adam(learning_rate=0.001)`
- Epochs: `200`, `verbose=0`
- Validation split: `0.1` (monitor convergence during training)

### Evaluation & output — main.py
```
Training model...
Model trained!

=== Evaluation on test set ===
MAE:  0.XX
RMSE: 0.XX

=== Sample Prediction ===
Age: 21 | Major: Data Science | Tool: ChatGPT-4o | ...
GPA Baseline: 3.20 → Predicted GPA Post-AI: X.XX
```

---

## Data Flow

```
CSV
 └─ data.py: drop ID → OHE categoricals → scale numerics → split 80/20
      └─ X_train, y_train → model.py: build Sequential → fit 200 epochs
           └─ trained model → main.py: evaluate on X_test → print MAE/RMSE
                                        preprocess sample → predict → print
```

---

## Out of scope
- Saving/loading model weights
- Plotting training loss curves
- Hyperparameter tuning
- Dropout or BatchNormalization layers
