import streamlit as st
import pandas as pd
import numpy as np
import joblib
import smtplib
from email.message import EmailMessage
import plotly.express as px
import plotly.graph_objects as go
import os
from pymongo import MongoClient
import io

# --- IMPORTANT: Configure your email and MongoDB connection here ---
SENDER_EMAIL = "dishcoveryhelp@gmail.com"
APP_PASSWORD = "reln tijr nsol ezds" # Replace with your generated Google App Password

# MongoDB Connection Details
MONGO_URI = "mongodb://localhost:27017/" # For local MongoDB
DB_NAME = "university_dashboard"
MENTORS_COLLECTION = 'mentors_data'
STUDENT_MASTER_COLLECTION = 'student_master_data'
RAW_DATA_COLLECTIONS = {
    'attendance': 'attendance_data',
    'assessments': 'assessments_data',
    'fees': 'fees_data'
}
PROCESSED_DATA_COLLECTION = 'predicted_data'

# --- Database Functions ---
@st.cache_resource
def get_database_client():
    """Initializes and caches the MongoDB client."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping') # Check connection
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB. Please check your MONGO_URI. Error: {e}")
        return None

def save_to_mongodb(df, collection_name):
    """Saves a DataFrame to a specified MongoDB collection."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[collection_name]
        collection.delete_many({}) # Clear existing data
        collection.insert_many(df.to_dict('records'))
        return True
    return False

def load_from_mongodb(collection_name, query={}):
    """Loads data from a MongoDB collection into a DataFrame."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[collection_name]
        data = list(collection.find(query, {'_id': 0}))
        if data:
            return pd.DataFrame(data)
    return pd.DataFrame()

def load_all_static_data_to_db():
    """Loads static CSVs into MongoDB if they don't exist in the database."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        if db[MENTORS_COLLECTION].count_documents({}) == 0:
            st.info("Loading mentor data from 'mentors.csv' for the first time...")
            mentors_df = pd.read_csv('mentors.csv')
            save_to_mongodb(mentors_df, MENTORS_COLLECTION)
        
        if db[STUDENT_MASTER_COLLECTION].count_documents({}) == 0:
            st.info("Loading student master data from 'student_master.csv' for the first time...")
            student_master_df = pd.read_csv('student_master.csv')
            save_to_mongodb(student_master_df, STUDENT_MASTER_COLLECTION)

# --- Data Processing and Prediction ---
def process_uploaded_files_and_save_to_db(attendance_file, assessments_file, fees_file, model):
    """Processes the uploaded files, runs predictions, and saves results to MongoDB."""
    try:
        # Load student master data from the database
        student_master = load_from_mongodb(STUDENT_MASTER_COLLECTION)
        if student_master.empty:
            st.error("Student master data not found in the database. Please ensure 'student_master.csv' is in the directory and reload the app.")
            return None
            
        # Read uploaded files
        attendance_df = pd.read_csv(attendance_file)
        assessments_df = pd.read_csv(assessments_file)
        fees_df = pd.read_csv(fees_file)

        # Check for required columns
        required_cols = {
            'Attendance CSV': ['StudentID', 'attendance'],
            'Assessments CSV': ['StudentID', 'marks', 'attempts'],
            'Fees CSV': ['StudentID', 'fees_due'],
            'student_master_data': ['StudentID', 'StudentName', 'Department', 'MentorID'],
        }
        for file_name, df, cols in [('Attendance CSV', attendance_df, required_cols['Attendance CSV']), 
                                    ('Assessments CSV', assessments_df, required_cols['Assessments CSV']), 
                                    ('Fees CSV', fees_df, required_cols['Fees CSV'])] :
            if not all(col in df.columns for col in cols):
                st.error(f"Error: The uploaded '{file_name}' is missing required columns: {', '.join([col for col in cols if col not in df.columns])}")
                return None
        
        # Merge the DataFrames
        fused_df = pd.merge(attendance_df, assessments_df, on='StudentID', how='outer')
        fused_df = pd.merge(fused_df, fees_df, on='StudentID', how='outer')
        final_df = pd.merge(fused_df, student_master, on='StudentID', how='outer')
        
        # Run prediction and classify risk levels
        features = ['attendance', 'marks', 'attempts', 'fees_due']
        df_for_prediction = final_df.dropna(subset=features).copy()
        
        predicted_df = final_df.copy()
        if not df_for_prediction.empty:
            predictions_proba = model.predict_proba(df_for_prediction[features])[:, 1]
            df_for_prediction['Dropout_Probability'] = predictions_proba
            predicted_df = pd.merge(final_df, df_for_prediction[['StudentID', 'Dropout_Probability']], on='StudentID', how='left')
        else:
            predicted_df['Dropout_Probability'] = np.nan

        def get_risk_level(prob):
            if pd.isna(prob): return "Not Predicted"
            elif prob >= 0.7: return '🔴 High Risk'
            elif prob >= 0.3: return '🟡 Moderate Risk'
            else: return '🟢 Low Risk'
        
        predicted_df['Risk_Level'] = predicted_df['Dropout_Probability'].apply(get_risk_level)

        # Save all data to MongoDB
        if save_to_mongodb(attendance_df, RAW_DATA_COLLECTIONS['attendance']) and \
           save_to_mongodb(assessments_df, RAW_DATA_COLLECTIONS['assessments']) and \
           save_to_mongodb(fees_df, RAW_DATA_COLLECTIONS['fees']) and \
           save_to_mongodb(predicted_df, PROCESSED_DATA_COLLECTION):
            return predicted_df
        else:
            st.error("Failed to save data to MongoDB.")
            return None
    
    except FileNotFoundError as e:
        st.error(f"An error occurred. Please ensure all required files are present. Error: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        return None

# --- Helper and Email Functions (Unchanged) ---
def get_mentor_email(mentor_id, mentors_df):
    try:
        email = mentors_df[mentors_df['MentorID'] == mentor_id]['Email'].iloc[0]
        return email
    except IndexError:
        return None

def send_notification(mentor_email, student_details):
    msg = EmailMessage()
    msg['Subject'] = f"Urgent: High-Risk Mentee Alert - Student {student_details['StudentID']}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = mentor_email
    attendance_str = f"{student_details['attendance']:.1f}%" if not pd.isna(student_details['attendance']) else "Not available"
    marks_str = f"{student_details['marks']:.1f}" if not pd.isna(student_details['marks']) else "Not available"
    fees_str = f"₹{student_details['fees_due']:.0f}" if not pd.isna(student_details['fees_due']) else "Not available"
    msg.set_content(f"""Dear Mentor,\n\nThis is an automated alert. Your mentee, {student_details['StudentID']}, has been flagged as high risk.\n\nSummary:\n- Dropout Probability: {student_details['Dropout_Probability']:.2%}\n- Attendance: {attendance_str}\n- Marks: {marks_str}\n- Fees Due: {fees_str}\n\nRegards,\nUniversity Admin""")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email to {mentor_email}: {e}")
        return False

# --- User Authentication and Session Management ---
def check_password(mentor_id, password, mentors_df):
    if mentor_id in mentors_df['MentorID'].values:
        user_row = mentors_df[mentors_df['MentorID'] == mentor_id]
        if user_row['Password'].iloc[0] == password:
            st.session_state['logged_in'] = True
            st.session_state['mentor_id'] = mentor_id
            st.session_state['department'] = user_row['Department'].iloc[0]
            st.session_state['page'] = 'dashboard'
            st.success("Login Successful! 🎉")
            st.rerun()
    st.error("Invalid Mentor ID or Password")
    return False

# --- Student Details View (Unchanged) ---
def show_student_details(df):
    student_id = st.session_state['selected_student_id']
    student_row = df[df['StudentID'] == student_id].iloc[0]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<h1 class="section-header">📋 Student Profile: {student_id}</h1>', unsafe_allow_html=True)
        st.markdown(f"**Name:** {student_row['StudentName']} | **Department:** {student_row['Department']}")
    with col2:
        if st.button("⬅️ Back to Dashboard", use_container_width=True, key="back_to_dashboard"):
            st.session_state['page'] = 'dashboard'
            st.session_state['selected_student_id'] = None
            st.rerun()
    
    st.markdown("---")
    
    dept_df = df[df['Department'] == student_row['Department']]
    dept_avg_attendance = dept_df['attendance'].mean()
    dept_avg_marks = dept_df['marks'].mean()
    
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if pd.isna(student_row['attendance']):
            st.metric(label="📅 Attendance", value="Not Available", delta=None)
        else:
            attendance_delta = student_row['attendance'] - dept_avg_attendance
            st.metric(label="📅 Attendance", value=f"{student_row['attendance']:.1f}%", delta=f"{attendance_delta:.1f}% vs Dept Avg" if not pd.isna(attendance_delta) else None)
    with col2:
        if pd.isna(student_row['marks']):
            st.metric(label="📝 Marks", value="Not Available", delta=None)
        else:
            marks_delta = student_row['marks'] - dept_avg_marks
            st.metric(label="📝 Marks", value=f"{student_row['marks']:.1f}", delta=f"{marks_delta:.1f} vs Dept Avg" if not pd.isna(marks_delta) else None)
    with col3:
        if pd.isna(student_row['attempts']):
            st.metric(label="✅ Attempts", value="Not Available")
        else:
            st.metric(label="✅ Attempts", value=f"{int(student_row['attempts'])}")
    with col4:
        if pd.isna(student_row['fees_due']):
            st.metric(label="💰 Fees Due", value="Not Available")
        else:
            st.metric(label="💰 Fees Due", value=f"₹{int(student_row['fees_due'])}")
    
    st.markdown("### 📈 Dropout Risk Analysis")
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        if pd.isna(student_row['Dropout_Probability']):
            st.info("No prediction data available for this student.")
        else:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=student_row['Dropout_Probability'], domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dropout Probability"},
                gauge={'axis': {'range': [0, 1]}, 'steps': [{'range': [0, 0.3], 'color': "green"}, {'range': [0.3, 0.7], 'color': "yellow"}, {'range': [0.7, 1], 'color': "red"}], 'bar': {'color': "white"}, 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': student_row['Dropout_Probability']}}
            ))
            st.plotly_chart(fig, use_container_width=True)
    with risk_col2:
        st.markdown(f"**Risk Level:** <span style='font-size: 1.5rem; font-weight: bold; color: {'green' if student_row['Risk_Level'] == '🟢 Low Risk' else 'orange' if student_row['Risk_Level'] == '🟡 Moderate Risk' else 'red'}'>{student_row['Risk_Level']}</span>", unsafe_allow_html=True)
        if student_row['Risk_Level'] == '🔴 High Risk':
            st.warning("⚠️ This student is at high risk of dropping out. Proactive intervention is highly recommended.")
        elif student_row['Risk_Level'] == '🟡 Moderate Risk':
            st.info("This student is at moderate risk. Monitor their progress closely.")
        else:
            st.success("This student is at low risk.")
        if st.session_state['mentor_id'] == 'ADM-001' and st.button("Send Email Alert to Mentor"):
            mentor_email = get_mentor_email(student_row['MentorID'], st.session_state['mentors_df'])
            if mentor_email:
                with st.spinner('Sending email...'):
                    if send_notification(mentor_email, student_row):
                        st.success("Email alert sent successfully!")
                    else:
                        st.error("Failed to send email. Check credentials and app password.")
            else:
                st.error("Mentor email not found.")

# --- Dashboard View ---
def show_dashboard(df, mentors_df):
    """Displays the main dashboard, handling admin vs mentor views."""
    mentor_id = st.session_state['mentor_id']
    
    # Only Admin sees file upload and email button in the sidebar
    if mentor_id == 'ADM-001':
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 Upload & Process Data")
        attendance_file = st.sidebar.file_uploader("Upload Attendance CSV", type=['csv'], key="attendance_uploader")
        assessments_file = st.sidebar.file_uploader("Upload Assessments CSV", type=['csv'], key="assessments_uploader")
        fees_file = st.sidebar.file_uploader("Upload Fees CSV", type=['csv'], key="fees_uploader")
        
        if st.sidebar.button("🔄 Process & Save to Database", use_container_width=True, key="process_button"):
            if attendance_file and assessments_file and fees_file:
                st.session_state['predicted_df'] = process_uploaded_files_and_save_to_db(attendance_file, assessments_file, fees_file, st.session_state['model'])
                if st.session_state['predicted_df'] is not None:
                    st.success("✅ Data processed and saved to database successfully!")
                    st.rerun()
            else:
                st.warning("Please upload all three CSV files.")

    # Check if the database data is available
    if df.empty or 'MentorID' not in df.columns:
        if mentor_id == 'ADM-001':
            st.info("Please upload and process the data files to view the dashboard.")
        else:
            st.info("Data is not available yet. Please wait for the administrator to upload and process the latest data.")
        return

    # Filter data based on user role
    if mentor_id == 'ADM-001':
        st.subheader("University-Wide Student Data")
        filtered_df = df.copy()
    else:
        st.subheader(f"Dashboard for Mentor: {mentors_df[mentors_df['MentorID'] == mentor_id]['MentorName'].iloc[0]}")
        filtered_df = df[df['MentorID'] == mentor_id].copy()

    if filtered_df.empty:
        st.info("No students are currently assigned to this mentor or department.")
        return
    
    st.markdown('<h2 class="section-header">🔍 Quick Student Search</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        search_id = st.text_input("Enter Student ID", placeholder="e.g., STD-0001", label_visibility="collapsed")
    with col2:
        if st.button("🔍 Search Student", use_container_width=True, key="search_student_btn"):
            if search_id in filtered_df['StudentID'].values:
                st.session_state['selected_student_id'] = search_id
                st.session_state['page'] = 'student_details'
                st.rerun()
            else:
                st.error("❌ Student ID not found in your assigned list.")
    st.markdown("---")
    
    st.markdown('<h2 class="section-header">📈 Visual Analytics Dashboard</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "⚠️ Risk Distribution", "🏢 Department Overview"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Marks vs. Dropout Risk")
            plot_df = filtered_df.dropna(subset=['marks', 'Dropout_Probability'])
            fig_marks = px.scatter(plot_df, x='marks', y='Dropout_Probability', color='Risk_Level', color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'}, hover_data=['StudentID', 'attendance', 'fees_due'], title='Academic Performance vs. Risk Level')
            fig_marks.update_layout(height=400)
            st.plotly_chart(fig_marks, use_container_width=True)
        with col2:
            st.markdown("##### Attendance vs. Dropout Risk")
            plot_df = filtered_df.dropna(subset=['attendance', 'Dropout_Probability'])
            fig_att = px.scatter(plot_df, x='attendance', y='Dropout_Probability', color='Risk_Level', color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'}, hover_data=['StudentID', 'marks', 'fees_due'], title='Attendance Pattern vs. Risk Level')
            fig_att.update_layout(height=400)
            st.plotly_chart(fig_att, use_container_width=True)
    with tab2:
        st.markdown("##### Overall Risk Distribution")
        risk_counts_df = filtered_df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0).reset_index()
        risk_counts_df.columns = ['Risk_Level', 'Count']
        fig_risk_dist = px.bar(risk_counts_df, x='Risk_Level', y='Count', color='Risk_Level', color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'}, title='Student Risk Category Distribution')
        fig_risk_dist.update_layout(height=500)
        st.plotly_chart(fig_risk_dist, use_container_width=True)
    with tab3:
        if mentor_id == 'ADM-001':
            st.markdown("##### Risk Distribution by Department")
            dept_risk_counts = filtered_df.groupby('Department')['Risk_Level'].value_counts().unstack().fillna(0)
            dept_risk_counts_long = dept_risk_counts.reset_index().melt(id_vars='Department', var_name='Risk_Level', value_name='Count')
            fig_dept_breakdown = px.bar(dept_risk_counts_long, x='Department', y='Count', color='Risk_Level', title='Department-wise Risk Analysis', color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
            fig_dept_breakdown.update_layout(height=500)
            st.plotly_chart(fig_dept_breakdown, use_container_width=True)
        else:
            st.info("📋 Department overview is available for administrators only.")

    st.markdown("---")
    st.markdown('<h2 class="section-header">📊 Student Risk Analytics</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🔴 High Risk Students", len(filtered_df[filtered_df['Risk_Level'] == '🔴 High Risk']))
    with col2:
        st.metric("🟡 Moderate Risk Students", len(filtered_df[filtered_df['Risk_Level'] == '🟡 Moderate Risk']))
    with col3:
        st.metric("🟢 Low Risk Students", len(filtered_df[filtered_df['Risk_Level'] == '🟢 Low Risk']))
    with col4:
        st.metric("📈 Average Attendance", f"{filtered_df['attendance'].mean():.1f}%")
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📋 Student Database")
    with col2:
        st.info("💡 Click any row for detailed metrics")
    
    display_df = filtered_df.copy()
    display_df['attendance_display'] = display_df['attendance'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "Not Available")
    display_df['marks_display'] = display_df['marks'].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "Not Available")
    display_df['fees_due_display'] = display_df['fees_due'].apply(lambda x: f"₹{x:.0f}" if not pd.isna(x) else "Not Available")
    display_df['Risk_Level_display'] = display_df['Risk_Level'].fillna("Not Predicted")
    display_df['Dropout_Probability_display'] = display_df['Dropout_Probability'].fillna(0)

    event = st.dataframe(
        display_df,
        column_order=['StudentID', 'StudentName', 'Department', 'MentorID', 'Risk_Level_display', 'Dropout_Probability_display', 'attendance_display', 'marks_display', 'fees_due_display'],
        column_config={'StudentID': st.column_config.Column(label='🆔 Student ID', width='medium'), 'StudentName': st.column_config.Column(label='👤 Name', width='large'), 'Department': st.column_config.Column(label='🏢 Department', width='medium'), 'MentorID': st.column_config.Column(label='👨‍🏫 Mentor ID', width='medium'), 'Risk_Level_display': st.column_config.Column(label='⚠️ Risk Level', width='medium'), 'Dropout_Probability_display': st.column_config.ProgressColumn(label='📊 Risk Score', help='Predicted dropout probability', format='%.1f%%', min_value=0, max_value=1, width='medium'), 'attendance_display': st.column_config.TextColumn(label='📅 Attendance', width='small'), 'marks_display': st.column_config.TextColumn(label='📝 Marks', width='small'), 'fees_due_display': st.column_config.TextColumn(label='💰 Fees Due', width='medium')},
        use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", height=600
    )
    
    if len(event.selection.rows) > 0:
        st.session_state['selected_student_id'] = filtered_df.iloc[event.selection.rows[0]]['StudentID']
        st.session_state['page'] = 'student_details'
        st.rerun()

    if mentor_id == 'ADM-001' and st.sidebar.button("📧 Send High-Risk Alerts", use_container_width=True, key="dashboard_alerts"):
        high_risk_students = filtered_df[filtered_df['Risk_Level'] == '🔴 High Risk']
        notifications_sent_to = []
        if not high_risk_students.empty:
            with st.spinner('Sending alerts...'):
                for index, row in high_risk_students.iterrows():
                    mentor_email = get_mentor_email(row['MentorID'], mentors_df)
                    if mentor_email and row['MentorID'] not in notifications_sent_to:
                        if send_notification(mentor_email, row):
                            notifications_sent_to.append(row['MentorID'])
                if notifications_sent_to:
                    st.success(f"Notification(s) sent to: {', '.join(notifications_sent_to)}")
                else:
                    st.info("No high-risk students found or alerts have already been sent to their mentors.")
        else:
            st.info("No high-risk students to send alerts for.")

# --- Main App Execution Flow ---
def main():
    st.set_page_config(layout="wide", page_title="University Dashboard", page_icon="🎓", initial_sidebar_state="expanded")
    st.markdown("""<style> .main-title{text-align: center;color: #1f4e79;font-size: 3rem;font-weight: bold;margin-bottom: 2rem;text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}[data-testid="metric-container"] {background-color: #f8f9fa;border: 1px solid #e9ecef;padding: 1rem;border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}.section-header {color: #495057;border-bottom: 3px solid #007bff;padding-bottom: 0.5rem;margin-bottom: 1.5rem;}.stButton button {background-color: #007bff;color: white;border-radius: 6px;border: none;padding: 0.5rem 1rem;font-weight: 500;transition: all 0.3s;}.stButton button:hover {background-color: #0056b3;transform: translateY(-1px);box-shadow: 0 4px 8px rgba(0,0,0,0.2);}.css-1d391kg {background-color: #f8f9fa;}#MainMenu, footer, header {visibility: hidden;}</style>""", unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🎓 University Dropout Risk Analytics Platform</h1>', unsafe_allow_html=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['page'] = 'login'
        st.session_state['selected_student_id'] = None
    
    # Load static data from MongoDB or local files if DB is empty
    load_all_static_data_to_db()
    
    try:
        mentors_df = load_from_mongodb(MENTORS_COLLECTION)
        model = joblib.load('dropout_prediction_model_final.pkl')
        st.session_state['mentors_df'] = mentors_df
        st.session_state['model'] = model
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}. Please ensure 'dropout_prediction_model_final.pkl' is in the same directory.")
        return

    if st.session_state['logged_in']:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 👋 Welcome!")
            st.markdown(f"**User:** `{st.session_state['mentor_id']}`")
            if st.session_state['mentor_id'] != 'ADM-001':
                mentor_name = st.session_state['mentors_df'][st.session_state['mentors_df']['MentorID'] == st.session_state['mentor_id']]['MentorName'].iloc[0]
                st.markdown(f"**Mentor:** {mentor_name}")
                st.markdown(f"**Department:** {st.session_state['department']}")
            else:
                st.markdown("**Role:** System Administrator")
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
                st.session_state.clear()
                st.rerun()

        if st.session_state['page'] == 'dashboard':
            predicted_df = load_from_mongodb(PROCESSED_DATA_COLLECTION)
            show_dashboard(predicted_df, st.session_state['mentors_df'])
        elif st.session_state['page'] == 'student_details':
            predicted_df = load_from_mongodb(PROCESSED_DATA_COLLECTION)
            if not predicted_df.empty:
                show_student_details(predicted_df)
            else:
                st.warning("No data is available to display student details.")
                st.session_state['page'] = 'dashboard'
                st.rerun()
    else:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<div style='text-align: center; background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border: 1px solid #e9ecef;'><h2>🔐 System Login</h2><p style='color: #6c757d;'>Enter your credentials to access the analytics platform</p></div>", unsafe_allow_html=True)
            with st.form("login_form", clear_on_submit=False):
                mentor_id = st.text_input("🆔 Mentor ID", placeholder="e.g., ADM-001")
                password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                if st.form_submit_button("🚀 Login", use_container_width=True):
                    check_password(mentor_id, password, st.session_state['mentors_df'])
            with st.expander("🔍 Demo Credentials"):
                st.markdown("**Admin:** `ADM-001` / `admin123`\n**Mentor:** `MNT-001` / `mentor123`")

if __name__ == '__main__':
    main()