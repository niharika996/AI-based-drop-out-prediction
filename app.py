import streamlit as st
import pandas as pd
import numpy as np
import joblib
import smtplib
from email.message import EmailMessage
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
import multiprocessing
import time
import sys
import json
import re

# =====================================================================================
# CONFIGURATION
# =====================================================================================

SENDER_EMAIL = "dishcoveryhelp@gmail.com"
APP_PASSWORD = "reln tijr nsol ezds"  # Replace with your generated Google App Password

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "university_dashboard"
MENTORS_COLLECTION = 'mentors_data'
STUDENT_MASTER_COLLECTION = 'student_master_data'
PROCESSED_DATA_COLLECTION = 'predicted_data'

# =====================================================================================
# DATABASE & DATA PROCESSING FUNCTIONS
# =====================================================================================

@st.cache_resource
def get_database_client():
    """Initializes and caches the MongoDB client."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Fatal: Could not connect to MongoDB. Please ensure it's running. Error: {e}")
        return None

def get_role_from_id(mentor_id):
    """Derives role from MentorID prefix."""
    if mentor_id.startswith('ADM'): return 'Admin'
    if mentor_id.startswith('PRN'): return 'Principal'
    if mentor_id.startswith('HOD'): return 'HOD'
    # Be more inclusive for Mentors, covering MTR- and MNT- prefixes
    if mentor_id.startswith('MNT') or mentor_id.startswith('MTR'): return 'Mentor'
    return 'Mentor' # Default role


def load_from_mongodb(collection_name, query={}):
    """Loads data from a MongoDB collection into a DataFrame."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[collection_name]
        data = list(collection.find(query, {'_id': 0}))
        if data:
            df = pd.DataFrame(data)
            
            # --- FIX: Ensure sensitive fields are clean strings and 'Role' exists for reliable comparison ---
            if collection_name == MENTORS_COLLECTION:
                
                # 1. Clean sensitive fields
                for col in ['MentorID', 'Password', 'Role', 'Department']:
                    if col in df.columns:
                        # Explicitly convert to string and remove leading/trailing whitespace
                        df[col] = df[col].astype(str).str.strip()

                # 2. Guarantee 'Role' column existence (FIX for KeyError)
                if 'Role' not in df.columns:
                    # If 'Role' is missing from the DB documents, derive it from MentorID
                    df['Role'] = df['MentorID'].apply(get_role_from_id)
            # --- END FIX ---
            
            return df
    return pd.DataFrame()

def save_to_mongodb(df, collection_name, week_number):
    """Saves a DataFrame to MongoDB, overwriting only for the specified week."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[collection_name]
        
        # Overwrite only for the specific week
        collection.delete_many({'Week': week_number})
        
        records_to_save = df.to_dict('records')
        if records_to_save:
            collection.insert_many(records_to_save)
            return len(records_to_save)
    return 0
    
def update_student_reason(student_id, week, note, mentor_id):
    """Updates a student's record with a mentor's note."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[PROCESSED_DATA_COLLECTION]
        query = {"StudentID": student_id, "Week": int(week)}
        update = {"$set": {"MentorNotes": note, "MentorID_Notes": mentor_id}}
        try:
            result = collection.update_one(query, update)
            return result.modified_count > 0
        except Exception as e:
            st.error(f"An error occurred while updating the document: {e}")
    return False

def setup_initial_db_data():
    """
    Loads static CSVs into MongoDB. 
    Crucially: It only loads mentors_data IF the collection is currently empty 
    to prevent overwriting existing data.
    """
    client = get_database_client()
    if not client: return

    db = client[DB_NAME]
    try:
        # --- FIX START: ONLY LOAD MENTORS DATA IF COLLECTION IS EMPTY ---
        mentor_collection = db[MENTORS_COLLECTION]
        if mentor_collection.count_documents({}) == 0:
            
            mentors_df = pd.read_csv('mentors.csv')
            
            # Aggressive data cleaning on all string columns to remove hidden characters/spaces
            string_cols = ['MentorID', 'MentorName', 'Department', 'Password', 'Email']
            for col in string_cols:
                if col in mentors_df.columns:
                    # Remove all whitespace and non-printable characters for sensitive fields like ID and Password
                    if col in ['MentorID', 'Password']:
                        mentors_df[col] = mentors_df[col].astype(str).apply(lambda x: re.sub(r'\s+', '', x).strip())
                    # Clean leading/trailing spaces for other fields
                    else:
                        mentors_df[col] = mentors_df[col].astype(str).str.strip()


            # Create a 'Role' column based on the MentorID prefix (handling common formats like MTR/MNT)
            mentors_df['Role'] = mentors_df['MentorID'].apply(get_role_from_id)
            
            # Ensure Role column is clean string before saving
            if 'Role' in mentors_df.columns:
                 mentors_df['Role'] = mentors_df['Role'].astype(str).str.strip()
            
            mentor_collection.insert_many(mentors_df.to_dict('records'))
            print("Mentors data loaded from CSV (initial load only).")
        # --- FIX END ---
        
        # Student master data is only loaded if the collection is empty
        if db[STUDENT_MASTER_COLLECTION].count_documents({}) == 0:
            student_master_df = pd.read_csv('student_master.csv')
            db[STUDENT_MASTER_COLLECTION].insert_many(student_master_df.to_dict('records'))
            print("Student master data loaded from CSV.")
            
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A required setup file is missing: {e}. Please add it to the directory and restart.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during initial data setup: {e}")
        st.stop()

def process_uploaded_files(attendance_file, assessments_file, fees_file, model, week_number):
    """Processes uploaded files, validates them, runs predictions, and saves results."""
    try:
        student_master = load_from_mongodb(STUDENT_MASTER_COLLECTION)
        if student_master.empty:
            st.error("Student master data not found in the database. Cannot process files.")
            return 0

        # --- ROBUST VALIDATION BLOCK ---
        attendance_df = pd.read_csv(attendance_file)
        if not all(col in attendance_df.columns for col in ['StudentID', 'attendance']):
            st.error("Attendance CSV is missing required columns: 'StudentID', 'attendance'.")
            return 0

        assessments_df = pd.read_csv(assessments_file)
        if not all(col in assessments_df.columns for col in ['StudentID', 'marks', 'attempts']):
            st.error("Assessments CSV is missing required columns: 'StudentID', 'marks', 'attempts'.")
            return 0

        fees_df = pd.read_csv(fees_file)
        if not all(col in fees_df.columns for col in ['StudentID', 'fees_due']):
            st.error("Fees CSV is missing required columns: 'StudentID', 'fees_due'.")
            return 0
        # --- END VALIDATION ---

        # Add 'Week' column
        for df in [attendance_df, assessments_df, fees_df]:
            df['Week'] = week_number

        # Save raw data to their respective collections
        save_to_mongodb(attendance_df, "attendance_data", week_number)
        save_to_mongodb(assessments_df, "assessments_data", week_number)
        save_to_mongodb(fees_df, "fees_data", week_number)

        # Merge for prediction
        fused_df = pd.merge(attendance_df, assessments_df, on=['StudentID', 'Week'], how='outer')
        fused_df = pd.merge(fused_df, fees_df, on=['StudentID', 'Week'], how='outer')
        final_df = pd.merge(fused_df, student_master, on='StudentID', how='left')

        features = ['attendance', 'marks', 'attempts', 'fees_due']
        df_for_prediction = final_df.dropna(subset=features).copy()
        
        if not df_for_prediction.empty:
            predictions_proba = model.predict_proba(df_for_prediction[features])[:, 1]
            df_for_prediction['Dropout_Probability'] = predictions_proba
            predicted_df = pd.merge(final_df, df_for_prediction[['StudentID', 'Week', 'Dropout_Probability']], on=['StudentID', 'Week'], how='left')
        else:
            predicted_df = final_df.copy()
            predicted_df['Dropout_Probability'] = np.nan

        def get_risk_level(prob):
            if pd.isna(prob): return "Not Predicted"
            if prob >= 0.7: return '🔴 High Risk'
            if prob >= 0.3: return '🟡 Moderate Risk'
            return '🟢 Low Risk'

        predicted_df['Risk_Level'] = predicted_df['Dropout_Probability'].apply(get_risk_level)
        
        return save_to_mongodb(predicted_df, PROCESSED_DATA_COLLECTION, week_number)
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        return 0


# =====================================================================================
# HELPER & EMAIL FUNCTIONS
# =====================================================================================
def get_mentor_email(mentor_id, mentors_df):
    try:
        return mentors_df.loc[mentors_df['MentorID'] == mentor_id, 'Email'].iloc[0]
    except IndexError:
        return None

def send_notification(mentor_email, student_details):
    msg = EmailMessage()
    msg['Subject'] = f"Urgent: High-Risk Mentee Alert - Student {student_details['StudentID']}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = mentor_email
    content = f"""Dear Mentor,
This is an automated alert. Your mentee, {student_details['StudentName']} ({student_details['StudentID']}), has been flagged as high risk.

Latest Weekly Metrics:
- Dropout Probability: {student_details['Dropout_Probability']:.2%}
- Attendance: {student_details.get('attendance', 'N/A'):.1f}%
- Marks: {student_details.get('marks', 'N/A'):.1f}

Please intervene as necessary.

Regards,
University Admin
"""
    msg.set_content(content)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email to {mentor_email}: {e}")
        return False

def _send_alerts_in_background(high_risk_students_json, mentors_df_json):
    """Background process to send emails without blocking the UI."""
    try:
        high_risk_students = pd.read_json(high_risk_students_json)
        mentors_df = pd.read_json(mentors_df_json)
        
        sent_count = 0
        for _, student_info in high_risk_students.iterrows():
            mentor_email = get_mentor_email(student_info['MentorID'], mentors_df)
            if mentor_email and send_notification(mentor_email, student_info):
                sent_count += 1
        return sent_count
    except Exception as e:
        print(f"Error in background email process: {e}", file=sys.stderr)
        return 0

# =====================================================================================
# UI & PAGE RENDERING
# =====================================================================================

def render_login_page(mentors_df):
    """Displays the login form."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; background-color: #f8f9fa; padding: 2rem; border-radius: 10px;'><h2>🔐 System Login</h2></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            mentor_id = st.text_input("🆔 User ID", placeholder="e.g., ADM-001")
            password = st.text_input("🔑 Password", type="password")
            if st.form_submit_button("🚀 Login", use_container_width=True):
                # Input cleaning matches the aggressive cleaning applied during setup
                clean_mentor_id = re.sub(r'\s+', '', mentor_id).strip()
                clean_password = re.sub(r'\s+', '', password).strip()
                
                # The data in mentors_df is now guaranteed to be clean due to the update in load_from_mongodb
                if clean_mentor_id in mentors_df['MentorID'].values:
                    user_row = mentors_df[mentors_df['MentorID'] == clean_mentor_id].iloc[0]
                    
                    if user_row['Password'] == clean_password:
                        st.session_state.logged_in = True
                        st.session_state.mentor_id = clean_mentor_id
                        st.session_state.user_role = user_row['Role']
                        st.session_state.user_dept = user_row['Department']
                        st.session_state.page = 'dashboard'
                        st.rerun()
                    else:
                        st.error("Invalid User ID or Password") # Failure on password match
                else:
                    st.error("Invalid User ID or Password") # Failure on ID existence
        with st.expander("🔍 Demo Credentials"):
            st.markdown("""
            **Admin:** `ADM-001` / `admin123`
            **Principal:** `PRN-001` / `principal123`
            **HOD (CSE):** `HOD-CSE` / `hodcse123`
            **Mentor (CSE):** `MTR-CSE-01` / `mentorpass`
            """)

def render_sidebar(latest_week_df, mentors_df):
    """Renders the sidebar for logged-in users."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 👋 Welcome!")
        st.markdown(f"**User:** `{st.session_state.mentor_id}`")
        user_info = mentors_df[mentors_df['MentorID'] == st.session_state.mentor_id].iloc[0]
        st.markdown(f"**Role:** {user_info['Role']}")
        if st.session_state.user_role != 'Admin':
            st.markdown(f"**Name:** {user_info['MentorName']}")
        if st.session_state.user_role in ['HOD', 'Mentor']:
            st.markdown(f"**Department:** {user_info['Department']}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, key="logout_button"):
            st.session_state.clear()
            st.rerun()

        # Admin-only features
        if st.session_state.user_role == 'Admin':
            st.markdown("### 📂 Upload Weekly Data")
            latest_week = st.session_state.get('latest_week', 0)
            with st.form("upload_form"):
                week_number = st.number_input("Enter Week Number", min_value=1, step=1, value=latest_week + 1)
                attendance_file = st.file_uploader("Upload Attendance CSV", type=['csv'])
                assessments_file = st.file_uploader("Upload Assessments CSV", type=['csv'])
                fees_file = st.file_uploader("Upload Fees CSV", type=['csv'])
                if st.form_submit_button("🔄 Process & Save", use_container_width=True):
                    if all([attendance_file, assessments_file, fees_file]):
                        records_saved = process_uploaded_files(attendance_file, assessments_file, fees_file, st.session_state.model, week_number)
                        if records_saved > 0:
                            st.success(f"✅ Data for Week {week_number} processed successfully! ({records_saved} records saved)")
                            st.session_state.latest_week = week_number
                            st.rerun()
                        else:
                            st.error("Failed to process files. Please check the file contents.")
                    else:
                        st.warning("Please upload all three CSV files.")

            st.markdown("---")
            st.markdown("### 📧 Alerts")
            if st.button("Send High-Risk Alerts", use_container_width=True, key="send_alerts_button"):
                high_risk_students = latest_week_df[latest_week_df['Risk_Level'] == '🔴 High Risk']
                if not high_risk_students.empty:
                    st.info("Sending alerts in the background. The app will remain responsive.")
                    
                    # Convert dataframes to JSON to pass to the new process
                    high_risk_json = high_risk_students.to_json()
                    mentors_df_json = mentors_df.to_json()
                    
                    # Start the process without waiting for it to finish
                    # FIX: Removed the colon after 'target'
                    p = multiprocessing.Process(target=_send_alerts_in_background, args=(high_risk_json, mentors_df_json))
                    p.start()
                else:
                    st.info("No high-risk students to send alerts for.")

        if st.session_state.user_role == 'Admin':
            st.markdown("---")
            with st.expander("🔍 Debug Information"):
                all_data = load_from_mongodb(PROCESSED_DATA_COLLECTION)
                total_records = len(all_data)
                st.write(f"Total records in DB: {total_records}")
                if not all_data.empty and 'Week' in all_data.columns:
                    latest_week = int(all_data['Week'].max())
                    latest_week_data = all_data[all_data['Week'] == latest_week]
                    st.write(f"Latest week found: {latest_week}")
                    st.write(f"Records for latest week: {len(latest_week_data)}")
                    if st.session_state.user_role in ['HOD', 'Mentor']:
                         filtered_records = len(latest_week_df)
                         st.write(f"Records on your dashboard: {filtered_records}")
                    else:
                         st.write(f"Records on your dashboard: {len(latest_week_data)}")

def render_dashboard_page(all_data, mentors_df):
    """Displays the main dashboard with data from the latest week."""
    
    # Filter data based on user role
    if st.session_state.user_role == 'Admin':
        st.subheader("University-Wide Data")
        if all_data.empty:
            st.info("No data available. Please upload files to begin.")
            return
        
        latest_week = int(all_data['Week'].max())
        st.session_state.latest_week = latest_week
        df = all_data[all_data['Week'] == latest_week].copy()

    elif st.session_state.user_role == 'Principal':
        st.subheader("University-Wide Data")
        if all_data.empty:
            st.info("No data available for your role.")
            return
        latest_week = int(all_data['Week'].max())
        st.session_state.latest_week = latest_week
        df = all_data[all_data['Week'] == latest_week].copy()

    elif st.session_state.user_role == 'HOD':
        st.subheader(f"Dashboard for Department: {st.session_state.user_dept}")
        if all_data.empty:
            st.info("No data available for your department.")
            return
        latest_week = int(all_data['Week'].max())
        st.session_state.latest_week = latest_week
        df = all_data[(all_data['Week'] == latest_week) & (all_data['Department'] == st.session_state.user_dept)].copy()
        if df.empty:
            st.info(f"No students found in the {st.session_state.user_dept} department for the latest week ({latest_week}).")
            return
            
    elif st.session_state.user_role == 'Mentor':
        mentor_info = mentors_df[mentors_df['MentorID'] == st.session_state.mentor_id].iloc[0]
        st.subheader(f"Dashboard for Mentor: {mentor_info['MentorName']}")
        if all_data.empty:
            st.info("No data available for your students.")
            return
        latest_week = int(all_data['Week'].max())
        st.session_state.latest_week = latest_week
        df = all_data[(all_data['Week'] == latest_week) & (all_data['MentorID'] == st.session_state.mentor_id)].copy()
        if df.empty:
            st.info(f"No students are assigned to you for the latest week ({latest_week}).")
            return

    # User-agnostic dashboard content
    st.markdown(f"**Displaying data for Week {latest_week}** ({len(df)} students)")
    
    # Search bar for students
    student_id_search = st.text_input("Search for a Student ID", placeholder="e.g., STD-0001")
    if st.button("Go to Student Page", key="search_button"):
        if student_id_search in df['StudentID'].values:
            st.session_state.selected_student_id = student_id_search
            st.session_state.page = 'student_details'
            st.rerun()
        else:
            st.error(f"Student ID '{student_id_search}' not found in the latest week's data.")

    # --- VISUALIZATIONS ---
    st.markdown('<h2 class="section-header">📈 Visual Analytics Dashboard</h2>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "⚠️ Risk Distribution", "🏢 Department Overview"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df.dropna(subset=['marks', 'Dropout_Probability']),
                             x='marks', y='Dropout_Probability', color='Risk_Level',
                             trendline='ols', title='Marks vs. Dropout Risk',
                             color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # FIX: Corrected the hex code from '#28a45' to '#28a745'
            fig = px.scatter(df.dropna(subset=['attendance', 'Dropout_Probability']),
                             x='attendance', y='Dropout_Probability', color='Risk_Level',
                             trendline='ols', title='Attendance vs. Dropout Risk',
                             color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        risk_counts = df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0)
        fig = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, title='Overall Student Risk Distribution',
                     color=risk_counts.index, color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if st.session_state.user_role in ['Admin', 'Principal']:
            metric = st.selectbox("Select metric:", ['attendance', 'marks'])
            fig = px.box(df.dropna(subset=[metric]), x='Department', y=metric, color='Department',
                         title=f'Distribution of {metric.capitalize()} by Department')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("This feature is available for Principals and Admins only.")

    # --- Student Database Table ---
    st.markdown("---")
    st.markdown("### 📋 Student Database (Latest Week)")
    st.info("💡 Click any row to see a student's detailed weekly progress.")
    event = st.dataframe(
        df,
        column_order=["StudentID", "StudentName", "Department", "MentorID", "Risk_Level", "Dropout_Probability", "MentorNotes"],
        column_config={
            'Dropout_Probability': st.column_config.ProgressColumn('Risk Score', format='%.2f', min_value=0, max_value=1),
            'MentorNotes': st.column_config.TextColumn("Mentor Notes")
        },
        height=500,
        on_select="rerun", selection_mode="single-row"
    )

    if event.selection.rows:
        st.session_state.selected_student_id = df.iloc[event.selection.rows[0]]['StudentID']
        st.session_state.page = 'student_details'
        st.rerun()

def render_student_details_page(all_data):
    """Displays the detailed historical view for a single student."""
    student_id = st.session_state.selected_student_id
    student_history = all_data[all_data['StudentID'] == student_id].sort_values(by='Week').reset_index()
    
    if student_history.empty:
        st.warning(f"No historical data found for student ID: {student_id}.")
        if st.button("⬅️ Back to Dashboard", use_container_width=True, key="back_button"):
            st.session_state.page = 'dashboard'; st.rerun()
        return

    latest_row = student_history.iloc[-1]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📋 Student Profile: {latest_row['StudentName']} ({student_id})")
    with col2:
        if st.button("⬅️ Back to Dashboard", use_container_width=True, key="back_button"):
            st.session_state.page = 'dashboard'; st.rerun()
    st.markdown("---")

    # Mentor Notes Section (visible only to Mentors and if a note exists for others)
    if st.session_state.user_role == 'Mentor' and latest_row['Risk_Level'] == '🔴 High Risk':
        st.subheader("📝 Add a Mentor Note")
        with st.form("mentor_note_form"):
            note_content = st.text_area("Reason for High Risk", value=latest_row.get('MentorNotes', ''))
            if st.form_submit_button("Save Note"):
                if update_student_reason(student_id, latest_row['Week'], note_content, st.session_state.mentor_id):
                    st.success("Note saved successfully!")
                    st.rerun()
                else:
                    st.error("Failed to save note.")
    
    if 'MentorNotes' in latest_row and pd.notna(latest_row['MentorNotes']):
        st.markdown(f"### 💬 Mentor Note")
        st.info(f"**Mentor ({latest_row.get('MentorID_Notes', 'N/A')}):** {latest_row['MentorNotes']}")

    # Trend Charts
    st.subheader("📈 Performance Trend Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(student_history, x='Week', y='attendance', title='Attendance Trend', markers=True, labels={'attendance': 'Attendance (%)'})
        fig.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(student_history, x='Week', y='marks', title='Marks Trend', markers=True)
        fig.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig, use_container_width=True)

    # Department Comparison and Gauge Chart
    st.subheader("📊 Latest Week Performance vs. Department Average")
    latest_week_dept_data = all_data[(all_data['Department'] == latest_row['Department']) & (all_data['Week'] == latest_row['Week'])]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Your Score'], y=[latest_row['attendance']], name='Your Score'))
        fig.add_trace(go.Bar(x=['Dept. Avg'], y=[latest_week_dept_data['attendance'].mean()], name='Dept. Avg'))
        fig.update_layout(title_text="Attendance Comparison", yaxis_title="Attendance (%)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Your Score'], y=[latest_row['marks']], name='Your Score'))
        fig.add_trace(go.Bar(x=['Dept. Avg'], y=[latest_week_dept_data['marks'].mean()], name='Dept. Avg'))
        fig.update_layout(title_text="Marks Comparison", yaxis_title="Marks")
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=latest_row['Dropout_Probability'],
            title={'text': "Dropout Probability"},
            gauge={'axis': {'range': [0, 1]}, 'steps': [
                {'range': [0, 0.3], 'color': "green"}, {'range': [0.3, 0.7], 'color': "yellow"}, {'range': [0.7, 1], 'color': "red"}],
                   'bar': {'color': 'white'}}))
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================================
# MAIN APPLICATION FLOW
# =====================================================================================

def main():
    st.set_page_config(layout="wide", page_title="University Dashboard", page_icon="🎓")
    st.markdown('<h1 style="text-align: center;">🎓 University Dropout Risk Analytics Platform</h1>', unsafe_allow_html=True)

    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    if 'page' not in st.session_state: st.session_state.page = 'login'

    # Load persistent resources once
    if 'model' not in st.session_state:
        try:
            st.session_state.model = joblib.load('dropout_prediction_model_final.pkl')
            setup_initial_db_data()
            st.session_state.mentors_df = load_from_mongodb(MENTORS_COLLECTION)
            if st.session_state.mentors_df.empty:
                 st.error("Fatal: Mentor data could not be loaded. App cannot start.")
                 st.stop()
        except FileNotFoundError:
            st.error("Fatal: 'dropout_prediction_model_final.pkl' not found. App cannot start.")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during startup: {e}")
            st.stop()

    # Page Routing
    if not st.session_state.logged_in:
        render_login_page(st.session_state.mentors_df)
    else:
        all_data = load_from_mongodb(PROCESSED_DATA_COLLECTION)
        
        # Determine the latest week for display
        latest_week_df = pd.DataFrame()
        if not all_data.empty and 'Week' in all_data.columns:
            latest_week = int(all_data['Week'].max())
            latest_week_df = all_data[all_data['Week'] == latest_week]
            st.session_state.latest_week = latest_week
        
        # Filter the latest week data based on user role
        user_role = st.session_state.user_role
        user_id = st.session_state.mentor_id
        
        # Use a copy of the department value for HOD/Mentor filtering
        user_dept = st.session_state.user_dept
        
        if user_role == 'Admin' or user_role == 'Principal':
            pass # All data for the latest week is used
        elif user_role == 'HOD':
            latest_week_df = latest_week_df[latest_week_df['Department'] == user_dept]
        elif user_role == 'Mentor':
            latest_week_df = latest_week_df[latest_week_df['MentorID'] == user_id]

        render_sidebar(latest_week_df, st.session_state.mentors_df)
        
        if st.session_state.page == 'dashboard':
            render_dashboard_page(all_data, st.session_state.mentors_df)
        elif st.session_state.page == 'student_details':
            render_student_details_page(all_data)

if __name__ == '__main__':
    # This check is crucial for multiprocessing to work correctly.
    # It prevents the child process from running the full Streamlit app.
    # The fix ensures reliable data comparison during login.
    main()
