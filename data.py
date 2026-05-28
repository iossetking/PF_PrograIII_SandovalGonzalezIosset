import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ['Major', 'Primary_AI_Tool', 'Main_Usage_Case', 'AI_Ethics_Concern']
NUMERIC_COLS = ['Age', 'Task_Frequency_Daily', 'GPA_Baseline', 'Time_Saved_Hours_Weekly', 'Career_Confidence_Score']
TARGET = 'GPA_Post_AI'

def load_data():
    df = pd.read_csv('AI_Impact_Student_Life_2026.csv')
    df = df.drop(columns=['Student_ID'])

    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)

    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    scaler = MinMaxScaler()
    X[NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])

    ohe_columns = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, ohe_columns
