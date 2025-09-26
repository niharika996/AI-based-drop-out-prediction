# 🎓 University Dropout Risk Analytics Platform

This Streamlit application provides a robust, role-based dashboard for university staff (Admins, Principals, HODs, and Mentors) to monitor student performance, predict dropout risk using a machine learning model, and facilitate timely intervention.

---

## ✨ Features

- **Role-Based Access**: Secure login for Admin, Principal, HOD, and Mentor roles, with tailored views and permissions.  
- **Dropout Prediction**: Utilizes a pre-trained Scikit-learn model (`dropout_prediction_model_final.pkl`) to calculate weekly dropout probability and assign risk levels (High, Moderate, Low).  
- **Weekly Data Upload (Admin Only)**: Admins can upload weekly student data (Attendance, Assessments, Fees) via CSV files.  
- **Real-time Alerts**: Admins can trigger background email notifications to Mentors for students flagged as High Risk.  
- **Data Visualization**: Interactive Plotly charts for performance trends, risk distribution, and departmental comparisons.  
- **Mentor Intervention**: Mentors can add descriptive notes for high-risk students to track intervention efforts.  
- **MongoDB Integration**: Uses MongoDB to store persistent data (mentor profiles, student masters, and weekly processed data).  

---

## 🛠️ Prerequisites

To run this application, you must have the following installed:

- **Python 3.8+**  
- **MongoDB Server**: Running locally (default connection string `mongodb://localhost:27017/`).  
- **Required Python Libraries** (listed below).  

---

## ⚙️ Installation and Setup

### 1. Install Dependencies

Install all necessary Python libraries using pip:

```bash
pip install streamlit pandas numpy scikit-learn plotly pymongo joblib
