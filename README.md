# GPA Post-AI Prediction

Neural network that predicts a student's GPA after adopting AI tools, trained on the **AI Impact on Student Life 2026** dataset (1,500 students).

## What it does

Given a student's profile — age, major, AI tool used, usage habits, baseline GPA, and ethics concern level — the model predicts their post-AI GPA.

The network is a 3-layer Dense architecture (`64 → 32 → 1`) trained with Mean Squared Error. Categorical features are One-Hot Encoded; numeric features are normalized with MinMaxScaler.

**Sample output:**
```
Training model...
Model trained!

--- Evaluation on test set ---
MAE:  0.1479
RMSE: 0.1851

--- Sample Prediction ---
Age: 21
Major: Data Science
Tool: ChatGPT-4o
Usage: Exam Prep
Frequency: 3/day
Time Saved: 5h/week
Ethics Concern: Medium
Career Confidence: 7/10
GPA Baseline: 3.20 → Predicted GPA Post-AI: 3.50
```

## Dataset

`AI_Impact_Student_Life_2026.csv` — 1,500 rows, 11 columns.

| Column | Type | Description |
|---|---|---|
| `Age` | int | Student age (18–25) |
| `Major` | category | Field of study (6 majors) |
| `Primary_AI_Tool` | category | Main AI tool used (5 tools) |
| `Task_Frequency_Daily` | int | Daily AI interactions |
| `Main_Usage_Case` | category | How AI is used (5 cases) |
| `GPA_Baseline` | float | GPA before AI adoption |
| `GPA_Post_AI` | float | GPA after AI adoption *(target)* |
| `Time_Saved_Hours_Weekly` | int | Hours saved per week |
| `AI_Ethics_Concern` | category | Low / Medium / High |
| `Career_Confidence_Score` | int | Confidence score (1–10) |

## Project structure

```
├── data.py       # Load, preprocess, and split the dataset
├── model.py      # Build and train the neural network
├── main.py       # Orchestrate training, evaluation, and prediction
└── AI_Impact_Student_Life_2026.csv
```

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
# Install dependencies
uv sync

# Run
uv run main.py
```
