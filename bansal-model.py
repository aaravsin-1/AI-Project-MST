import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_and_save_model(csv_path="loan_approval_data.csv"):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return None, None

    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # Clean column names (strip spaces and lowercase)
    df.columns = df.columns.str.strip().str.lower()

    # 2. SPACE-PROOF CLEANING
    # This removes the leading spaces found in your data (e.g., ' Graduate' -> 'Graduate')
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # 3. Explicit Encoding
    encoding = {
        'education': {'Graduate': 1, 'Not Graduate': 0},
        'self_employed': {'Yes': 1, 'No': 0},
        'loan_status': {'Approved': 1, 'Rejected': 0}
    }
    df.replace(encoding, inplace=True)

    # 4. Final Validation
    # Ensure no NaNs were created by failed mapping
    df = df.dropna()
    
    if len(df) == 0:
        print("❌ Error: 0 samples remaining. Check if CSV values match the encoding keys.")
        return None, None

    # Define Features and Target
    X = df.drop(columns=['loan_id', 'loan_status'], errors='ignore')
    y = df['loan_status'].astype(int)

    # 5. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 7. Evaluation & Feature Importance
    y_pred = model.predict(X_test_scaled)
    print(f"✅ Training Successful! Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    
    print("\n--- Feature Importance (Ranked) ---")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances)

    # 8. Save Assets
    joblib.dump(model, "loan_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "features.pkl")
    return model, scaler

def get_prediction():
    try:
        model = joblib.load("loan_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return

    print("\n--- Enter Applicant Details ---")
    user_data = []
    for f in features:
        val = input(f"{f.replace('_', ' ').title()} (Value): ")
        user_data.append(float(val))

    user_df = pd.DataFrame([user_data], columns=features)
    user_scaled = scaler.transform(user_df)
    
    prediction = model.predict(user_scaled)[0]
    result = "✅ LOAN APPROVED" if prediction == 1 else "❌ LOAN REJECTED"
    print(f"\nResult: {result}")

if __name__ == "__main__":
    m, s = train_and_save_model()
    if m is not None:
        while True:
            get_prediction()
            if input("\nPredict another? (y/n): ").lower() != 'y':
                break