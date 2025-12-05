import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model():
    print("🏥 Loading Medical Records...")
    try:
        df = pd.read_csv('data/diabetes.csv')
    except FileNotFoundError:
        print("Error: 'data/diabetes.csv' not found. Please download from Kaggle.")
        exit()

    # --- STEP 1: PREPROCESSING (The "Pro" move) ---
    # In this dataset, values like Glucose=0 or BP=0 are errors. 
    # We replace them with the column average (mean).
    clean_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in clean_cols:
        # Replace 0 with NaN (Not a Number), then fill with Mean
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].mean())

    # --- STEP 2: SPLIT FEATURES & TARGET ---
    X = df.drop('Outcome', axis=1) # Features (Vitals)
    y = df['Outcome']              # Target (1=Diabetic, 0=Healthy)

    # --- STEP 3: SCALING (Crucial for KNN) ---
    # Insulin ranges 0-800, Age is 20-80. We must scale them to 0-1 range.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- STEP 4: TRAIN KNN ---
    # Split data: 80% for training history, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    k = 11  # K=11 is often good for binary classification (avoids ties)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Check accuracy
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ System Initialized. Accuracy: {acc*100:.2f}%")
    
    return knn, scaler

def asha_worker_tool(model, scaler):
    print("\n--- 👩‍⚕️ RURAL HEALTH TRIAGE INTERFACE 👨‍⚕️ ---")
    print("Enter patient vitals below:")

    try:
        # Collecting inputs (Standard medical metrics)
        pregnancies = float(input("Number of Pregnancies (e.g., 2): "))
        glucose = float(input("Glucose Level (e.g., 120): "))
        bp = float(input("Blood Pressure (e.g., 70): "))
        skin = float(input("Skin Thickness (mm) (e.g., 20): "))
        insulin = float(input("Insulin Level (e.g., 79): "))
        bmi = float(input("BMI (e.g., 32.0): "))
        pedigree = float(input("Diabetes Pedigree Function (e.g., 0.5): "))
        age = float(input("Age (e.g., 33): "))

        # Prepare data for prediction
        patient_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age]])
        
        # WE MUST SCALE THE INPUT just like we scaled the training data
        patient_data_scaled = scaler.transform(patient_data)

        # Predict
        prediction = model.predict(patient_data_scaled)
        probability = model.predict_proba(patient_data_scaled)[0][1] * 100

        print("\n---------------- RESULTS ----------------")
        if prediction[0] == 1:
            print(f"⚠️ DIAGNOSIS: HIGH RISK (Diabetic)")
            print(f"🚑 ACTION: Refer to City Hospital immediately.")
        else:
            print(f"✅ DIAGNOSIS: HEALTHY")
            print(f"📝 ACTION: Routine checkup recommended.")
        
        print(f"🔍 Confidence Score: {probability:.2f}%")
        print("-----------------------------------------")

    except ValueError:
        print("❌ Invalid Input. Please enter numbers only.")

if __name__ == "__main__":
    # Train once
    model, scaler = train_model()

    # Loop for multiple patients
    while True:
        asha_worker_tool(model, scaler)
        cont = input("\nCheck next patient? (y/n): ")
        if cont.lower() != 'y':
            break