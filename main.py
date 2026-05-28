import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import model
import data

CATEGORICAL_COLS = ['Major', 'Primary_AI_Tool', 'Main_Usage_Case', 'AI_Ethics_Concern']
NUMERIC_COLS = ['Age', 'Task_Frequency_Daily', 'GPA_Baseline', 'Time_Saved_Hours_Weekly', 'Career_Confidence_Score']

def evaluate(trained_model, X_test, y_test):
    predictions = trained_model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("\n--- Evaluation on test set ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

def predict_sample(trained_model, scaler, ohe_columns):
    sample = {
        'Age': 21,
        'Major': 'Data Science',
        'Primary_AI_Tool': 'ChatGPT-4o',
        'Task_Frequency_Daily': 3,
        'Main_Usage_Case': 'Exam Prep',
        'GPA_Baseline': 3.2,
        'Time_Saved_Hours_Weekly': 5,
        'AI_Ethics_Concern': 'Medium',
        'Career_Confidence_Score': 7
    }

    df = pd.DataFrame([sample])
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)
    df = df.reindex(columns=ohe_columns, fill_value=0)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    result = trained_model.predict(df.values, verbose=0)[0][0]

    print("\n--- Sample Prediction ---")
    print(f"Age: {sample['Age']}")
    print(f"Major: {sample['Major']}")
    print(f"Tool: {sample['Primary_AI_Tool']}")
    print(f"Usage: {sample['Main_Usage_Case']}")
    print(f"Frequency: {sample['Task_Frequency_Daily']}/day")
    print(f"Time Saved: {sample['Time_Saved_Hours_Weekly']}h/week")
    print(f"Ethics Concern: {sample['AI_Ethics_Concern']}")
    print(f"Career Confidence: {sample['Career_Confidence_Score']}/10")
    print(f"GPA Baseline: {sample['GPA_Baseline']:.2f} → Predicted GPA Post-AI: {result:.2f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler, ohe_columns = data.load_data()
    trained_model, history = model.train_model(X_train, y_train)
    evaluate(trained_model, X_test, y_test)
    predict_sample(trained_model, scaler, ohe_columns)
