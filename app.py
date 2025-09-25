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
ATTENDANCE_COLLECTION = 'attendance_data'
ASSESSMENTS_COLLECTION = 'assessments_data'
FEES_COLLECTION = 'fees_data'

# =====================================================================================
# DATABASE & DATA PROCESSING FUNCTIONS
# =====================================================================================

@st.cache_resource
def get_database_client():
    """Initializes and caches the MongoDB client."""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Fatal: Could not connect to MongoDB. Please ensure it's running. Error: {e}")
        return None

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

def save_to_mongodb(df, collection_name, week_number):
    """Saves a DataFrame to MongoDB, overwriting only for the specified week."""
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[collection_name]
        # This is the key part: it deletes all documents for a specific week before inserting new ones.
        # This prevents duplicate records for the same week while allowing the collection to grow with new weeks.
        collection.delete_many({'Week': int(week_number)}) # Fixed: Convert to int
        collection.insert_many(df.to_dict('records'))
        return True
    return False

def update_student_reason(student_id, week, reason, mentor_id):
    """
    Updates a specific student's record with a reason provided by a mentor.
    This function adds a 'MentorNotes' and 'MentorID_Notes' field to the document.
    """
    client = get_database_client()
    if client:
        db = client[DB_NAME]
        collection = db[PROCESSED_DATA_COLLECTION]
        query = {'StudentID': student_id, 'Week': int(week)} # Fixed: Convert to int
        update = {'$set': {'MentorNotes': reason, 'MentorID_Notes': mentor_id}}
        result = collection.update_one(query, update)
        return result.modified_count > 0
    return False

def setup_initial_db_data():
    """Loads static CSVs into MongoDB if collections are empty."""
    client = get_database_client()
    if not client: return

    db = client[DB_NAME]
    try:
        if db[MENTORS_COLLECTION].count_documents({}) == 0:
            mentors_df = pd.read_csv('mentors.csv')
            db[MENTORS_COLLECTION].insert_many(mentors_df.to_dict('records'))
        if db[STUDENT_MASTER_COLLECTION].count_documents({}) == 0:
            student_master_df = pd.read_csv('student_master.csv')
            db[STUDENT_MASTER_COLLECTION].insert_many(student_master_df.to_dict('records'))
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
            return False

        # --- ROBUST VALIDATION BLOCK ---
        attendance_df = pd.read_csv(attendance_file)
        if not all(col in attendance_df.columns for col in ['StudentID', 'attendance']):
            st.error("Attendance CSV is missing required columns: 'StudentID', 'attendance'.")
            return False

        assessments_df = pd.read_csv(assessments_file)
        if not all(col in assessments_df.columns for col in ['StudentID', 'marks', 'attempts']):
            st.error("Assessments CSV is missing required columns: 'StudentID', 'marks', 'attempts'.")
            return False

        fees_df = pd.read_csv(fees_file)
        if not all(col in fees_df.columns for col in ['StudentID', 'fees_due']):
            st.error("Fees CSV is missing required columns: 'StudentID', 'fees_due'.")
            return False
        # --- END VALIDATION ---

        for df in [attendance_df, assessments_df, fees_df]:
            df['Week'] = week_number
        
        # Save individual dataframes to their own collections
        save_to_mongodb(attendance_df, ATTENDANCE_COLLECTION, week_number)
        save_to_mongodb(assessments_df, ASSESSMENTS_COLLECTION, week_number)
        save_to_mongodb(fees_df, FEES_COLLECTION, week_number)

        fused_df = pd.merge(attendance_df, assessments_df, on=['StudentID', 'Week'], how='outer')
        fused_df = pd.merge(fused_df, fees_df, on=['StudentID', 'Week'], how='outer')
        final_df = pd.merge(fused_df, student_master, on='StudentID', how='left')

        features = ['attendance', 'marks', 'attempts', 'fees_due']
        df_for_prediction = final_df.dropna(subset=features).copy()

        if df_for_prediction.empty:
            st.warning("The uploaded data contains too many missing values to generate predictions. Please check your files.")
            predicted_df = final_df.copy()
            predicted_df['Dropout_Probability'] = np.nan
        else:
            predictions_proba = model.predict_proba(df_for_prediction[features])[:, 1]
            df_for_prediction['Dropout_Probability'] = predictions_proba
            predicted_df = pd.merge(final_df, df_for_prediction[['StudentID', 'Week', 'Dropout_Probability']], on=['StudentID', 'Week'], how='left')
            
        def get_risk_level(prob):
            if pd.isna(prob): return "Not Predicted"
            if prob >= 0.7: return '🔴 High Risk'
            if prob >= 0.3: return '🟡 Moderate Risk'
            return '🟢 Low Risk'

        predicted_df['Risk_Level'] = predicted_df['Dropout_Probability'].apply(get_risk_level)
        
        if save_to_mongodb(predicted_df, PROCESSED_DATA_COLLECTION, week_number):
            client = get_database_client()
            if client:
                db = client[DB_NAME]
                count = db[PROCESSED_DATA_COLLECTION].count_documents({'Week': week_number})
                if count > 0:
                    st.success(f"✅ Data for Week {week_number} processed successfully! There are **{count}** records in the database.")
                    return True
                else:
                    st.error(f"❌ Data processing failed. The database is empty after the save operation.")
                    return False
        else:
            st.error("❌ Failed to save data to the database. Please check your MongoDB connection.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")
        return False


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
    except Exception:
        return False

# Function to run in a separate process
def _send_alerts_in_background(sender_email, app_password, mentors_df_json, high_risk_students_json):
    """
    Function to send alerts in a separate process.
    Takes JSON serialized dataframes as arguments.
    """
    mentors_df = pd.read_json(mentors_df_json)
    high_risk_students = pd.read_json(high_risk_students_json)

    mentors_to_notify = high_risk_students['MentorID'].unique()
    for mentor_id in mentors_to_notify:
        mentor_email = get_mentor_email(mentor_id, mentors_df)
        student_info = high_risk_students[high_risk_students['MentorID'] == mentor_id].iloc[0]
        if mentor_email:
            # We don't need a return value, just send the email
            send_notification(mentor_email, student_info)

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
            mentor_id = st.text_input("🆔 Mentor ID", placeholder="e.g., ADM-001")
            password = st.text_input("🔑 Password", type="password")
            if st.form_submit_button("🚀 Login", use_container_width=True):
                if mentor_id in mentors_df['MentorID'].values:
                    user_row = mentors_df[mentors_df['MentorID'] == mentor_id].iloc[0]
                    if user_row['Password'] == password:
                        st.session_state.logged_in = True
                        st.session_state.mentor_id = mentor_id
                        st.session_state.page = 'dashboard'
                        st.rerun()
                st.error("Invalid Mentor ID or Password")
        with st.expander("🔍 Demo Credentials"):
            st.markdown("**Admin:** `ADM-001` / `admin123`\n**Mentor:** `MNT-001` / `mentorpass`")

def render_sidebar(latest_week_df, mentors_df):
    """Renders the sidebar for logged-in users."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 👋 Welcome!")
        st.markdown(f"**User:** `{st.session_state.mentor_id}`")
        user_info = mentors_df[mentors_df['MentorID'] == st.session_state.mentor_id].iloc[0]
        if st.session_state.mentor_id == 'ADM-001':
            st.markdown("**Role:** System Administrator")
        else:
            st.markdown(f"**Mentor:** {user_info['MentorName']}")
            st.markdown(f"**Department:** {user_info['Department']}")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True, key="logout_button"):
            st.session_state.clear()
            st.rerun()

        # Admin-only features
        if st.session_state.mentor_id == 'ADM-001':
            st.markdown("### 📂 Upload Weekly Data")
            with st.form("upload_form"):
                latest_week = st.session_state.get('latest_week', 0)
                week_number = st.number_input("Enter Week Number", min_value=1, step=1, value=latest_week + 1)
                attendance_file = st.file_uploader("Upload Attendance CSV", type=['csv'])
                assessments_file = st.file_uploader("Upload Assessments CSV", type=['csv'])
                fees_file = st.file_uploader("Upload Fees CSV", type=['csv'])
                if st.form_submit_button("🔄 Process & Save", use_container_width=True):
                    if all([attendance_file, assessments_file, fees_file]):
                        if process_uploaded_files(attendance_file, assessments_file, fees_file, st.session_state.model, week_number):
                           st.rerun()
                    else:
                        st.warning("Please upload all three CSV files.")

            st.markdown("---")
            st.markdown("### 📧 Alerts")
            if st.button("Send High-Risk Alerts", use_container_width=True, key="send_alerts_button"):
                high_risk_students = latest_week_df[latest_week_df['Risk_Level'] == '🔴 High Risk']
                if not high_risk_students.empty:
                    # Serialize dataframes to JSON for multiprocessing
                    mentors_df_json = mentors_df.to_json()
                    high_risk_students_json = high_risk_students.to_json()

                    # Start the background process without blocking the UI
                    process = multiprocessing.Process(target=_send_alerts_in_background, args=(SENDER_EMAIL, APP_PASSWORD, mentors_df_json, high_risk_students_json))
                    process.daemon = True # Allows process to terminate if parent app exits
                    process.start()
                    st.info("✅ Alerts are being sent in the background. The app will remain responsive.")
                else:
                    st.info("No high-risk students to send alerts for.")
            
            # New Debug Information Expander for Admins
            st.markdown("---")
            with st.expander("🛠️ Debug Information"):
                all_data = load_from_mongodb(PROCESSED_DATA_COLLECTION)
                st.info(f"Total records in DB: **{len(all_data)}**")
                if not all_data.empty and 'Week' in all_data.columns:
                    latest_week = int(all_data['Week'].max())
                    latest_week_df_debug = all_data[all_data['Week'] == latest_week]
                    st.info(f"Latest week found: **{latest_week}**")
                    st.info(f"Records for latest week: **{len(latest_week_df_debug)}**")
                    
                    if st.session_state.mentor_id == 'ADM-001':
                        filtered_records = latest_week_df_debug
                    else:
                        filtered_records = latest_week_df_debug[latest_week_df_debug['MentorID'] == st.session_state.mentor_id]
                    st.info(f"Records on your dashboard: **{len(filtered_records)}**")
                else:
                    st.info("No data or 'Week' column found in database.")


def render_dashboard_page(all_data, mentors_df):
    """Displays the main dashboard with data from the latest week."""
    if all_data.empty or 'Week' not in all_data.columns:
        st.info("The dashboard is currently empty. Please upload weekly data using the sidebar menu to view visualizations and student data.")
        return

    latest_week = int(all_data['Week'].max())
    st.session_state.latest_week = latest_week
    # Filter the entire dataset to get only the latest week's data
    latest_week_data = all_data[all_data['Week'] == latest_week].copy()

    # Filter data based on user role
    if st.session_state.mentor_id == 'ADM-001':
        st.subheader(f"University-Wide Data (Latest Week: {latest_week})")
        filtered_df = latest_week_data
    else:
        mentor_info = mentors_df[mentors_df['MentorID'] == st.session_state.mentor_id].iloc[0]
        st.subheader(f"Dashboard for Mentor: {mentor_info['MentorName']} (Latest Week: {latest_week})")
        # Filter the latest week's data by the mentor's ID
        filtered_df = latest_week_data[latest_week_data['MentorID'] == st.session_state.mentor_id].copy()

    if filtered_df.empty:
        st.info("No student data is available for the latest week. This may be because no data has been processed or your assigned students are not in this week's data.")
        return

    # --- Student Search Bar ---
    st.markdown("---")
    st.markdown("### 🔍 Search Student")
    with st.form("search_form"):
        student_id_search = st.text_input("Enter Student ID to view details")
        if st.form_submit_button("Search"):
            if student_id_search and student_id_search in filtered_df['StudentID'].values:
                st.session_state.selected_student_id = student_id_search
                st.session_state.page = 'student_details'
                st.rerun()
            else:
                st.error("Student ID not found in the current week's data. Please check the ID or try a different one.")
    
    st.markdown("---")

    # --- VISUALIZATIONS ---
    st.markdown('<h2 class="section-header">📈 Visual Analytics Dashboard</h2>', unsafe_allow_html=True)
    st.info(f"The visualizations below display data for the **{len(filtered_df)}** students from **Week {latest_week}**.")
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "⚠️ Risk Distribution", "🏢 Department Overview"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Use the filtered_df which only contains the latest week's data
            plot_df = filtered_df.dropna(subset=['marks', 'Dropout_Probability'])
            if plot_df.empty:
                st.warning("No data with both marks and risk scores to display this chart.")
            else:
                fig = px.scatter(plot_df, x='marks', y='Dropout_Probability', color='Risk_Level',
                                 trendline='ols', title='Marks vs. Dropout Risk',
                                 color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Use the filtered_df which only contains the latest week's data
            plot_df = filtered_df.dropna(subset=['attendance', 'Dropout_Probability'])
            if plot_df.empty:
                st.warning("No data with both attendance and risk scores to display this chart.")
            else:
                fig = px.scatter(plot_df, x='attendance', y='Dropout_Probability', color='Risk_Level',
                                 trendline='ols', title='Attendance vs. Dropout Risk',
                                 color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
                st.plotly_chart(fig, use_container_width=True)
    with tab2:
        # Use the filtered_df which only contains the latest week's data
        risk_counts = filtered_df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0)
        if risk_counts.sum() == 0:
             st.warning("No risk data to display this chart.")
        else:
            fig = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, title='Overall Student Risk Distribution',
                         color=risk_counts.index, color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'})
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        if st.session_state.mentor_id == 'ADM-001':
            metric = st.selectbox("Select metric:", ['attendance', 'marks'])
            plot_df = filtered_df.dropna(subset=[metric])
            if plot_df.empty:
                 st.warning(f"No {metric} data to display this chart.")
            else:
                fig = px.box(plot_df, x='Department', y=metric, color='Department',
                             title=f'Distribution of {metric.capitalize()} by Department')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Department overview is available for administrators only.")

    # --- Student Database Table ---
    st.markdown("---")
    st.markdown("### 📋 Student Database (Latest Week)")
    st.info("💡 Click any row to see a student's detailed weekly progress.")
    
    # Render dataframe with the new MentorNotes column
    event = st.dataframe(
        filtered_df,
        column_config={
            'Dropout_Probability': st.column_config.ProgressColumn('Risk Score', format='%.2f', min_value=0, max_value=1),
            'MentorNotes': st.column_config.Column('Mentor Notes', help='Notes added by the mentor')
        },
        height=500, # Increased height to show more rows at once
        on_select="rerun", selection_mode="single-row",
        use_container_width=True,
    )
    if event.selection.rows:
        st.session_state.selected_student_id = filtered_df.iloc[event.selection.rows[0]]['StudentID']
        st.session_state.page = 'student_details'
        st.rerun()

def render_student_details_page(all_data):
    """Displays the detailed historical view for a single student."""
    student_id = st.session_state.selected_student_id
    student_history = all_data[all_data['StudentID'] == student_id].sort_values(by='Week').reset_index()
    latest_row = student_history.iloc[-1]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(f"📋 Student Profile: {latest_row['StudentName']} ({student_id})")
    with col2:
        if st.button("⬅️ Back to Dashboard", use_container_width=True, key="back_button"):
            st.session_state.page = 'dashboard'; st.rerun()
    st.markdown("---")

    # Mentor Notes Section - visible for all users if notes exist
    if 'MentorNotes' in latest_row and pd.notna(latest_row['MentorNotes']):
        st.info(f"**Notes from Mentor ({latest_row.get('MentorID_Notes', 'N/A')}):** {latest_row['MentorNotes']}")
    
    # Form for mentors to add notes (only for non-admins and high-risk students)
    if st.session_state.mentor_id != 'ADM-001' and latest_row['Risk_Level'] == '🔴 High Risk':
        st.markdown("### 📝 Add a Note for This Student")
        with st.form("mentor_notes_form", clear_on_submit=True):
            note = st.text_area("Reason for Student Risk (e.g., family issues, health problems, etc.)", max_chars=500)
            if st.form_submit_button("Save Note"):
                if note:
                    if update_student_reason(student_id, int(latest_row['Week']), note, st.session_state.mentor_id): # Fixed: Convert to int
                        st.success("✅ Note saved successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save the note. Please try again.")
                else:
                    st.warning("Please enter a note before saving.")

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

    # Page Routing
    if not st.session_state.logged_in:
        render_login_page(st.session_state.mentors_df)
    else:
        all_data = load_from_mongodb(PROCESSED_DATA_COLLECTION)
        latest_week_df = pd.DataFrame()
        if not all_data.empty and 'Week' in all_data.columns:
            latest_week = int(all_data['Week'].max())
            latest_week_df = all_data[all_data['Week'] == latest_week]
        
        render_sidebar(latest_week_df, st.session_state.mentors_df)
        
        if st.session_state.page == 'dashboard':
            render_dashboard_page(all_data, st.session_state.mentors_df)
        elif st.session_state.page == 'student_details':
            render_student_details_page(all_data)

if __name__ == '__main__':
    main()
