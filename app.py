import streamlit as st
import pandas as pd
import numpy as np
import joblib
import smtplib
from email.message import EmailMessage
import plotly.express as px
import plotly.graph_objects as go

# --- IMPORTANT: Configure your email sender details here ---
# You must generate a Google "App Password" to use this feature.
SENDER_EMAIL = "dishcoveryhelp@gmail.com"
APP_PASSWORD = "reln tijr nsol ezds" # This is NOT your email password

# --- Data Loading and Fusion ---
# This function simulates pulling data from separate sources and merging it.
@st.cache_data
def load_and_prepare_data():
    """Loads and merges all data files into a single DataFrame."""
    try:
        # Step 1: Read the raw, disaggregated data
        attendance_df = pd.read_csv('attendance.csv')
        assessments_df = pd.read_csv('assessments.csv')
        fees_df = pd.read_csv('fees.csv')
        
        # Step 2: Merge the DataFrames based on StudentID
        fused_df = pd.merge(attendance_df, assessments_df, on='StudentID', how='left')
        fused_df = pd.merge(fused_df, fees_df, on='StudentID', how='left')
        
        # Step 3: Load the master data files
        student_master = pd.read_csv('student_master.csv')
        mentors_df = pd.read_csv('mentors.csv')
        
        # Step 4: Merge fused data with master data for roles and departments
        final_df = pd.merge(fused_df, student_master, on='StudentID', how='left')
        
        # Step 5: Load the pre-trained model
        model = joblib.load('dropout_prediction_model_final.pkl')
        
        return final_df, mentors_df, model
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please ensure all data and model files are in the directory. Missing file: {e}")
        return None, None, None

# --- Helper function to get mentor email ---
def get_mentor_email(mentor_id, mentors_df):
    """Retrieves mentor's email address from the mentors DataFrame."""
    try:
        email = mentors_df[mentors_df['MentorID'] == mentor_id]['Email'].iloc[0]
        return email
    except IndexError:
        return None

# --- Email Notification Function (no UI output) ---
def send_notification(mentor_email, student_details):
    """Sends an email notification to a mentor about a high-risk student."""
    msg = EmailMessage()
    msg['Subject'] = f"Urgent: High-Risk Mentee Alert - Student {student_details['StudentID']}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = mentor_email
    msg.set_content(f"""
    Dear Mentor,

    This is an automated alert from the University Dropout Risk Prediction System.

    Your mentee, {student_details['StudentID']}, has been flagged as high risk.
    
    Here is a summary of their current status:
    - Dropout Probability: {student_details['Dropout_Probability']:.2%}
    - Attendance: {student_details['attendance']:.1f}%
    - Marks: {student_details['marks']:.1f}
    - Fees Due: ₹{student_details['fees_due']:.0f}
    
    Please log in to the dashboard for more details and consider a proactive intervention.
    
    Regards,
    University Admin
    """)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email to {mentor_email}: {e}") # Log error to console
        return False

# --- User Authentication and Session Management ---
def check_password(mentor_id, password, mentors_df):
    """Authenticates the user and sets up the session."""
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

# --- Student Details View ---
def show_student_details(df):
    """Displays a detailed report for a single student."""
    st.subheader(f"Student Report: {st.session_state['selected_student_id']}")
    
    # Get student data
    student_id = st.session_state['selected_student_id']
    student_row = df[df['StudentID'] == student_id].iloc[0]
    
    # Get departmental averages for comparison
    dept_df = df[df['Department'] == student_row['Department']]
    dept_avg_attendance = dept_df['attendance'].mean()
    dept_avg_marks = dept_df['marks'].mean()
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Attendance", value=f"{student_row['attendance']:.1f}%", delta=f"{student_row['attendance'] - dept_avg_attendance:.1f}% vs Dept Avg")
    with col2:
        st.metric(label="Marks", value=f"{student_row['marks']:.1f}", delta=f"{student_row['marks'] - dept_avg_marks:.1f} vs Dept Avg")
    with col3:
        st.metric(label="Fees Due", value=f"₹{student_row['fees_due']:.0f}")

    st.markdown('---')
    st.header("Visualizations for this Student")
    
    # Bar chart for performance comparison
    st.subheader("Performance vs. Departmental Average")
    performance_data = {
        'Metric': ['Attendance', 'Marks'],
        'Student Value': [student_row['attendance'], student_row['marks']],
        'Department Average': [dept_avg_attendance, dept_avg_marks]
    }
    performance_df = pd.DataFrame(performance_data)
    fig = go.Figure(data=[
        go.Bar(name='Student Value', x=performance_df['Metric'], y=performance_df['Student Value']),
        go.Bar(name='Department Average', x=performance_df['Metric'], y=performance_df['Department Average'])
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # Risk Meter (Gauge Chart)
    st.subheader("Dropout Risk Meter")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=student_row['Dropout_Probability'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Dropout Probability"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 30], 'color': "green"},
                   {'range': [30, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "red"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
    st.plotly_chart(fig_gauge)

    if st.button("⬅️ Back to Dashboard"):
        st.session_state['page'] = 'dashboard'
        st.session_state['selected_student_id'] = None
        st.rerun()

# --- Dashboard View Logic ---
def show_dashboard(df, mentors_df):
    """Displays the main dashboard with student data and visualizations."""
    mentor_id = st.session_state['mentor_id']
    
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
    
    # --- Student Search/Navigation ---
    st.markdown('---')
    st.header("Search for a Specific Student")
    search_id = st.text_input("Enter Student ID (e.g., STD-0001)")
    if st.button("View Student Details"):
        if search_id in filtered_df['StudentID'].values:
            st.session_state['selected_student_id'] = search_id
            st.session_state['page'] = 'student_details'
            st.rerun()
        else:
            st.error("Student ID not found in your assigned list. Please check the ID and try again.")
    st.markdown('---')

    # --- VISUALIZATION SECTION ---
    st.header("Visual Insights")
    
    # Scatter plot of Marks vs. Dropout Probability
    st.subheader("Marks vs. Dropout Risk")
    fig_marks = px.scatter(
        filtered_df, 
        x='marks', 
        y='Dropout_Probability', 
        color='Risk_Level',
        color_discrete_map={'🔴 High Risk':'red', '🟡 Moderate Risk':'orange', '🟢 Low Risk':'green'},
        hover_data=['StudentID', 'attendance', 'fees_due'],
        title='Marks vs. Predicted Dropout Probability'
    )
    st.plotly_chart(fig_marks, use_container_width=True)

    # Scatter plot of Attendance vs. Dropout Probability
    st.subheader("Attendance vs. Dropout Risk")
    fig_att = px.scatter(
        filtered_df, 
        x='attendance', 
        y='Dropout_Probability', 
        color='Risk_Level',
        color_discrete_map={'🔴 High Risk':'red', '🟡 Moderate Risk':'orange', '🟢 Low Risk':'green'},
        hover_data=['StudentID', 'marks', 'fees_due'],
        title='Attendance vs. Predicted Dropout Probability'
    )
    st.plotly_chart(fig_att, use_container_width=True)

    # Overall Risk Distribution (Admin and Mentor View)
    st.subheader("Overall Risk Distribution")
    risk_counts_df = filtered_df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0).reset_index()
    risk_counts_df.columns = ['Risk_Level', 'Count']
    fig_risk_dist = px.bar(
        risk_counts_df,
        x='Risk_Level',
        y='Count',
        color='Risk_Level',
        color_discrete_map={'🔴 High Risk':'red', '🟡 Moderate Risk':'orange', '🟢 Low Risk':'green'},
        title='Risk Distribution'
    )
    st.plotly_chart(fig_risk_dist, use_container_width=True)
    
    # Departmental Breakdown for Admin
    if mentor_id == 'ADM-001':
        st.subheader("Risk Distribution by Department")
        dept_risk_counts = filtered_df.groupby('Department')['Risk_Level'].value_counts().unstack().fillna(0)
        
        dept_risk_counts_long = dept_risk_counts.reset_index().melt(
            id_vars='Department', 
            var_name='Risk_Level', 
            value_name='Count'
        )

        fig_dept_breakdown = px.bar(
            dept_risk_counts_long,
            x='Department',
            y='Count',
            color='Risk_Level',
            title='Departmental Risk Breakdown',
            color_discrete_map={'🔴 High Risk':'red', '🟡 Moderate Risk':'orange', '🟢 Low Risk':'green'}
        )
        st.plotly_chart(fig_dept_breakdown, use_container_width=True)

    # --- Display Data Table with clickable rows ---
    st.markdown('---')
    st.subheader("Student Risk Table")
    st.info("💡 Click on any row to view detailed student metrics")
    
    # Create a copy with row selection
    display_df = filtered_df[['StudentID', 'StudentName', 'Department', 'MentorID', 'Risk_Level', 'Dropout_Probability', 'attendance', 'marks', 'fees_due']].reset_index(drop=True)
    
    # Use data_editor with selection enabled
    event = st.dataframe(
        display_df,
        column_config={
            'StudentID': st.column_config.Column(label='Student ID', width='small'),
            'StudentName': st.column_config.Column(label='Name', width='medium'),
            'MentorID': st.column_config.Column(label='Mentor ID'),
            'Risk_Level': st.column_config.Column(label='Risk Level'),
            'Dropout_Probability': st.column_config.ProgressColumn(
                label='Probability',
                help='Predicted dropout probability',
                format='%.2f%%',
                min_value=0,
                max_value=1
            ),
            'attendance': st.column_config.NumberColumn(label='Attendance', format='%.1f%%'),
            'marks': st.column_config.NumberColumn(label='Marks', format='%.1f'),
            'fees_due': st.column_config.NumberColumn(label='Fees Due', format='₹%.0f'),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # Check if a row was selected
    if len(event.selection.rows) > 0:
        selected_row_index = event.selection.rows[0]
        selected_student_id = display_df.iloc[selected_row_index]['StudentID']
        st.session_state['selected_student_id'] = selected_student_id
        st.session_state['page'] = 'student_details'
        st.rerun()

    # Add a button to send notifications
    if st.sidebar.button("Send High-Risk Alerts"):
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
                    mentors_str = ', '.join(notifications_sent_to)
                    st.success(f"Notification(s) sent to: {mentors_str}")
                else:
                    st.info("No high-risk students found or alerts have already been sent to their mentors.")
        else:
            st.info("No high-risk students to send alerts for.")


# --- Main App Execution Flow ---
def main():
    """Main application logic for the Streamlit app."""
    st.set_page_config(layout="wide", page_title="University Dashboard")
    st.title('University Dropout Risk Prediction')
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['page'] = 'login'
        st.session_state['selected_student_id'] = None
    
    df, mentors_df, model = load_and_prepare_data()
    if df is None or mentors_df is None or model is None:
        return

    # Main navigation logic
    if st.session_state['logged_in']:
        st.sidebar.title(f"Welcome, {st.session_state['mentor_id']}!")
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            # Clear cached data to ensure a fresh start for the next login
            if 'predicted_df' in st.session_state:
                del st.session_state['predicted_df']
            st.rerun()
        
        # --- Optimization: Run predictions only once per session ---
        if 'predicted_df' not in st.session_state:
            with st.spinner('Preparing dashboard... This may take a moment.'):
                features = ['attendance', 'marks', 'attempts', 'fees_due']
                predictions_proba = model.predict_proba(df[features])[:, 1]
                df['Dropout_Probability'] = predictions_proba
                def get_risk_level(prob):
                    if prob >= 0.7: return '🔴 High Risk'
                    elif prob >= 0.3: return '🟡 Moderate Risk'
                    else: return '🟢 Low Risk'
                df['Risk_Level'] = df['Dropout_Probability'].apply(get_risk_level)
                st.session_state['predicted_df'] = df

        if st.session_state['page'] == 'dashboard':
            show_dashboard(st.session_state['predicted_df'], mentors_df)

        elif st.session_state['page'] == 'student_details':
            show_student_details(st.session_state['predicted_df'])
    else:
        st.subheader("Login to your account")
        mentor_id = st.text_input("Mentor ID")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            check_password(mentor_id, password, mentors_df)

if __name__ == '__main__':
    main()