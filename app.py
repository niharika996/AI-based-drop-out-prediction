import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_fusion import fuse_data  # CORRECT: We now import the fusion function

# --- Load data and model (using st.cache_data for performance) ---
@st.cache_data
def load_and_prepare_data():
    try:
        # CORRECT: Fuse the live data from the raw CSVs
        df = fuse_data()
        
        # Load mentor credentials and student-mentor mapping
        student_master = pd.read_csv('student_master.csv')
        mentors_df = pd.read_csv('mentors.csv')
        
        # Merge the fused data with the master mapping
        df = pd.merge(df, student_master, on='StudentID', how='left')
        
        # Load the pre-trained model
        model = joblib.load('dropout_prediction_model_final.pkl')
        
        return df, mentors_df, model
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure all data and model files are in the directory. Missing file: {e}")
        return None, None, None

# --- User Authentication and Session Management ---
def check_password(mentor_id, password, mentors_df):
    if mentor_id in mentors_df['MentorID'].values:
        user_row = mentors_df[mentors_df['MentorID'] == mentor_id]
        if user_row['Password'].iloc[0] == password:
            st.session_state['logged_in'] = True
            st.session_state['mentor_id'] = mentor_id
            st.session_state['department'] = user_row['Department'].iloc[0]
            st.success("Login Successful! 🎉")
            st.rerun()
    st.error("Invalid Mentor ID or Password")
    return False

# --- Dashboard View Logic ---
def show_dashboard(df, model):
    mentor_id = st.session_state['mentor_id']

    if mentor_id == 'ADM-001':
        st.subheader("University-Wide Student Data")
        filtered_df = df.copy()
    else:
        st.subheader(f"Dashboard for Mentor: {mentor_id}")
        filtered_df = df[df['MentorID'] == mentor_id].copy()
    
    if filtered_df.empty:
        st.info("No students are currently assigned to this mentor or department.")
        return

    # Make predictions and calculate risk levels
    features = ['attendance', 'marks', 'attempts', 'fees_due']
    predictions_proba = model.predict_proba(filtered_df[features])[:, 1]
    filtered_df['Dropout_Probability'] = predictions_proba
    
    def get_risk_level(prob):
        if prob >= 0.7: return '🔴 High Risk'
        elif prob >= 0.3: return '🟡 Moderate Risk'
        else: return '🟢 Low Risk'

    filtered_df['Risk_Level'] = filtered_df['Dropout_Probability'].apply(get_risk_level)

    # --- Visualizations ---
    st.markdown('---')
    st.subheader("Risk Distribution")
    risk_counts = filtered_df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0)
    st.bar_chart(risk_counts)
    
    if mentor_id == 'ADM-001':
        st.markdown('---')
        st.subheader("Risk Distribution by Department")
        dept_risk_counts = filtered_df.groupby('Department')['Risk_Level'].value_counts().unstack().fillna(0)
        st.bar_chart(dept_risk_counts)

    # --- Display Data Table ---
    st.markdown('---')
    st.subheader("Student Risk Table")
    display_cols = ['StudentID', 'Department', 'MentorID', 'Risk_Level', 'Dropout_Probability', 'attendance', 'marks', 'attempts', 'fees_due']
    st.dataframe(filtered_df[display_cols].style.format({
        'Dropout_Probability': '{:.2%}',
        'attendance': '{:.1f}',
        'marks': '{:.1f}',
        'fees_due': '₹{:,.0f}'
    }), use_container_width=True)

# --- Main App Execution Flow ---
def main():
    st.set_page_config(layout="wide", page_title="University Dashboard")
    st.title('University Dropout Risk Prediction')
    st.markdown('### Secure Access Portal')

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    df, mentors_df, model = load_and_prepare_data()
    if df is None or mentors_df is None or model is None:
        return

    if not st.session_state['logged_in']:
        st.subheader("Login to your account")
        mentor_id = st.text_input("Mentor ID")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            check_password(mentor_id, password, mentors_df)
    else:
        st.sidebar.title(f"Welcome, {st.session_state['mentor_id']}!")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()
        show_dashboard(df, model)

if __name__ == '__main__':
    main()