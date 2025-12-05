# Gram-Swasthya-AI
# 🏥 Rural Health Triage System

### 👩‍⚕️ AI-Assisted Diagnostics for Rural Healthcare

**Domain:** Healthcare / Social Good  
**Tech Stack:** Python, Scikit-Learn, Pandas, Numpy

---

## 📖 Project Overview
In many rural areas, the ratio of doctors to patients is critically low. Villagers often ignore early symptoms of chronic diseases like Diabetes, leading to severe complications later.

This project bridges this gap by providing a **Smart Triage Tool** for frontline health workers (like ASHA workers). It uses the **K-Nearest Neighbors (KNN)** algorithm to analyze patient vitals and predict the likelihood of Diabetes, allowing for early intervention and prioritized referrals to city hospitals.
## screenshoot <img width="1248" height="640" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/16304a74-779f-4730-89ad-32ad6e7ef016" />

## 🎯 Societal & Government Impact
* **Empowering Frontline Workers:** Gives advanced diagnostic capability to non-specialist staff.
* **Reducing Hospital Load:** Prevents overcrowding in city hospitals by filtering low-risk patients.
* **Early Detection:** Identifies "At-Risk" patients before they develop critical complications.

## 🧠 Technical Implementation

### The Algorithm: K-Nearest Neighbors (KNN)
I chose KNN (K=11) because medical diagnosis is often comparative. If a patient's vitals (Glucose, BMI, Age) are mathematically similar to 11 other patients who had diabetes, it is highly probable this patient shares the condition.

### Data Engineering (Critical Step)
Real-world medical data is often messy. The dataset contained invalid zero values (e.g., Blood Pressure = 0).
* **Data Imputation:** I implemented a cleaning pipeline that detects these biological impossibilities and replaces them with the statistical mean of that column.
* **Feature Scaling:** Since *Insulin* (0-800) and *Age* (20-80) have different scales, I used `StandardScaler` to normalize all features, ensuring the distance calculation is accurate.

## 📂 Project Structure
```text
Rural_Health_Triage/
│
├── data/
│   └── diabetes.csv      # Pima Indians Diabetes Database
│
├── src/
│   └── app.py            # Main Application (Training + ASHA Interface)
│
├── requirements.txt      # Dependencies
└── README.md             # Documentation
How to Run Locally
Clone the Repository

Bash

git clone [https://github.com/YOUR-USERNAME/Rural-Health-Triage-System.git](https://github.com/YOUR-USERNAME/Rural-Health-Triage-System.git)
cd Rural-Health-Triage-System
Install Dependencies

Bash

pip install -r requirements.txt
Setup Data

Download the Pima Indians Diabetes Database.

Place the diabetes.csv file inside the data/ folder.

Run the Diagnostic Tool

Bash

python src/app.py
📊 Results
Algorithm: K-Nearest Neighbors (K=11)

Accuracy: ~75-80% (on test data)

Features: Pregnancies, Glucose, BP, Skin Thickness, Insulin, BMI, Pedigree Function, Age.

🔮 Future Scope
Deploying as a mobile app for tablets used by ASHA workers.

Adding support for Heart Disease prediction.

Multilingual interface (Hindi/Regional languages) for ease of use.
