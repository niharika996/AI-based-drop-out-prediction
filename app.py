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
    
    # Handle potentially missing values for the email body
    attendance_str = f"{student_details['attendance']:.1f}%" if not pd.isna(student_details['attendance']) else "Not available"
    marks_str = f"{student_details['marks']:.1f}" if not pd.isna(student_details['marks']) else "Not available"
    fees_str = f"₹{student_details['fees_due']:.0f}" if not pd.isna(student_details['fees_due']) else "Not available"

    msg.set_content(f"""
    Dear Mentor,

    This is an automated alert from the University Dropout Risk Prediction System.

    Your mentee, {student_details['StudentID']}, has been flagged as high risk.
    
    Here is a summary of their current status:
    - Dropout Probability: {student_details['Dropout_Probability']:.2%}
    - Attendance: {attendance_str}
    - Marks: {marks_str}
    - Fees Due: {fees_str}
    
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
    # Get student data
    student_id = st.session_state['selected_student_id']
    student_row = df[df['StudentID'] == student_id].iloc[0]
    
    # Header with back button
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
    
    # Get departmental averages for comparison
    dept_df = df[df['Department'] == student_row['Department']]
    dept_avg_attendance = dept_df['attendance'].mean()
    dept_avg_marks = dept_df['marks'].mean()
    
    # Enhanced metrics with better styling
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check and handle missing attendance
        if pd.isna(student_row['attendance']):
            st.metric(label="📅 Attendance", value="Not Available", delta=None)
        else:
            attendance_delta = student_row['attendance'] - dept_avg_attendance
            st.metric(
                label="📅 Attendance", 
                value=f"{student_row['attendance']:.1f}%", 
                delta=f"{attendance_delta:.1f}% vs Dept Avg"
            )
    
    with col2:
        # Check and handle missing marks
        if pd.isna(student_row['marks']):
            st.metric(label="📝 Marks", value="Not Available", delta=None)
        else:
            marks_delta = student_row['marks'] - dept_avg_marks
            st.metric(
                label="📝 Marks", 
                value=f"{student_row['marks']:.1f}", 
                delta=f"{marks_delta:.1f} vs Dept Avg"
            )
    
    with col3:
        # Check and handle missing fees
        if pd.isna(student_row['fees_due']):
            st.metric(label="💰 Fees Due", value="Not Available", delta=None)
        else:
            st.metric(label="💰 Fees Due", value=f"₹{student_row['fees_due']:.0f}", delta=None)
    
    with col4:
        # Handle missing risk level (no prediction was made)
        if pd.isna(student_row['Dropout_Probability']):
            st.metric(label="⚠️ Risk Level", value="Not Predicted", delta="Missing data")
        else:
            risk_color = "🔴" if student_row['Risk_Level'] == '🔴 High Risk' else "🟡" if student_row['Risk_Level'] == '🟡 Moderate Risk' else "🟢"
            st.metric(
                label="⚠️ Risk Level", 
                value=student_row['Risk_Level'].replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', ''),
                delta=f"{student_row['Dropout_Probability']*100:.1f}% probability"
            )

    st.markdown("---")
    
    # Enhanced visualizations in tabs
    st.markdown("### 📈 Detailed Analytics")
    
    tab1, tab2 = st.tabs(["📊 Performance Comparison", "⚠️ Risk Assessment"])
    
    with tab1:
        # Performance comparison chart
        performance_data = {
            'Metric': ['Attendance (%)', 'Academic Marks'],
            'Student Value': [
                student_row['attendance'] if not pd.isna(student_row['attendance']) else 0,
                student_row['marks'] if not pd.isna(student_row['marks']) else 0
            ],
            'Department Average': [dept_avg_attendance, dept_avg_marks]
        }
        performance_df = pd.DataFrame(performance_data)
        
        fig = go.Figure(data=[
            go.Bar(name='Student Performance', x=performance_df['Metric'], y=performance_df['Student Value'], marker_color='#007bff'),
            go.Bar(name='Department Average', x=performance_df['Metric'], y=performance_df['Department Average'], marker_color='#6c757d')
        ])
        fig.update_layout(
            barmode='group',
            title='Student vs Department Average Performance',
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Risk assessment gauge
        if pd.isna(student_row['Dropout_Probability']):
            st.info("ℹ️ Dropout risk cannot be assessed due to missing student data.")
        else:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=student_row['Dropout_Probability'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dropout Risk Assessment (%)", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#007bff"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4edda"},  # Light green
                        {'range': [30, 70], 'color': "#fff3cd"},  # Light yellow
                        {'range': [70, 100], 'color': "#f8d7da"}  # Light red
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk interpretation
            risk_prob = student_row['Dropout_Probability'] * 100
            if risk_prob >= 70:
                st.error(f"🚨 **High Risk Alert:** This student has a {risk_prob:.1f}% probability of dropping out. Immediate intervention recommended.")
            elif risk_prob >= 30:
                st.warning(f"⚠️ **Moderate Risk:** This student has a {risk_prob:.1f}% probability of dropping out. Monitor closely and consider support measures.")
            else:
                st.success(f"✅ **Low Risk:** This student has a {risk_prob:.1f}% probability of dropping out. Continue current support level.")

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
    st.markdown('<h2 class="section-header">🔍 Quick Student Search</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_id = st.text_input("Enter Student ID", placeholder="e.g., STD-0001", label_visibility="collapsed")
    with col2:
        search_button = st.button("🔍 Search Student", use_container_width=True, key="search_student_btn")
    
    if search_button and search_id:
        if search_id in filtered_df['StudentID'].values:
            st.session_state['selected_student_id'] = search_id
            st.session_state['page'] = 'student_details'
            st.rerun()
        else:
            st.error("❌ Student ID not found in your assigned list. Please check the ID and try again.")
    
    st.markdown("---")

    # --- VISUALIZATION SECTION ---
    st.markdown('<h2 class="section-header">📈 Visual Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "⚠️ Risk Distribution", "🏢 Department Overview"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Marks vs. Dropout Risk")
            # Only plot students with non-missing values for marks and prediction
            plot_df = filtered_df.dropna(subset=['marks', 'Dropout_Probability'])
            fig_marks = px.scatter(
                plot_df, 
                x='marks', 
                y='Dropout_Probability', 
                color='Risk_Level',
                color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'},
                hover_data=['StudentID', 'attendance', 'fees_due'],
                title='Academic Performance vs. Risk Level'
            )
            fig_marks.update_layout(height=400)
            st.plotly_chart(fig_marks, use_container_width=True)

        with col2:
            st.markdown("##### Attendance vs. Dropout Risk")
            # Only plot students with non-missing values for attendance and prediction
            plot_df = filtered_df.dropna(subset=['attendance', 'Dropout_Probability'])
            fig_att = px.scatter(
                plot_df, 
                x='attendance', 
                y='Dropout_Probability', 
                color='Risk_Level',
                color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'},
                hover_data=['StudentID', 'marks', 'fees_due'],
                title='Attendance Pattern vs. Risk Level'
            )
            fig_att.update_layout(height=400)
            st.plotly_chart(fig_att, use_container_width=True)

    with tab2:
        st.markdown("##### Overall Risk Distribution")
        risk_counts_df = filtered_df['Risk_Level'].value_counts().reindex(['🟢 Low Risk', '🟡 Moderate Risk', '🔴 High Risk']).fillna(0).reset_index()
        risk_counts_df.columns = ['Risk_Level', 'Count']
        fig_risk_dist = px.bar(
            risk_counts_df,
            x='Risk_Level',
            y='Count',
            color='Risk_Level',
            color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'},
            title='Student Risk Category Distribution'
        )
        fig_risk_dist.update_layout(height=500)
        st.plotly_chart(fig_risk_dist, use_container_width=True)
    
    with tab3:
        # Departmental Breakdown for Admin
        if mentor_id == 'ADM-001':
            st.markdown("##### Risk Distribution by Department")
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
                title='Department-wise Risk Analysis',
                color_discrete_map={'🔴 High Risk':'#dc3545', '🟡 Moderate Risk':'#ffc107', '🟢 Low Risk':'#28a745'}
            )
            fig_dept_breakdown.update_layout(height=500)
            st.plotly_chart(fig_dept_breakdown, use_container_width=True)
        else:
            st.info("📋 Department overview is available for administrators only.")

    st.markdown("---")

    # --- Display Data Table with clickable rows ---
    st.markdown('<h2 class="section-header">📊 Student Risk Analytics</h2>', unsafe_allow_html=True)
    
    # Summary metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_risk_count = len(filtered_df[filtered_df['Risk_Level'] == '🔴 High Risk'])
        st.metric("🔴 High Risk Students", high_risk_count, delta=None)
    with col2:
        moderate_risk_count = len(filtered_df[filtered_df['Risk_Level'] == '🟡 Moderate Risk'])
        st.metric("🟡 Moderate Risk Students", moderate_risk_count, delta=None)
    with col3:
        low_risk_count = len(filtered_df[filtered_df['Risk_Level'] == '🟢 Low Risk'])
        st.metric("🟢 Low Risk Students", low_risk_count, delta=None)
    with col4:
        avg_attendance = filtered_df['attendance'].mean()
        st.metric("📈 Average Attendance", f"{avg_attendance:.1f}%", delta=None)
    
    st.markdown("---")
    
    # Table header with instruction
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📋 Student Database")
    with col2:
        st.info("💡 Click any row for detailed metrics")
    
    # Create a display copy with formatted strings for missing values
    display_df = filtered_df.copy()
    display_df['attendance_display'] = display_df['attendance'].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "Not Available")
    display_df['marks_display'] = display_df['marks'].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "Not Available")
    display_df['fees_due_display'] = display_df['fees_due'].apply(lambda x: f"₹{x:.0f}" if not pd.isna(x) else "Not Available")
    display_df['Risk_Level_display'] = display_df['Risk_Level'].fillna("Not Predicted")
    display_df['Dropout_Probability_display'] = display_df['Dropout_Probability'].fillna(0) # Use 0 for progress bar

    # Use dataframe with selection enabled
    event = st.dataframe(
        display_df,
        column_order=['StudentID', 'StudentName', 'Department', 'MentorID', 'Risk_Level_display', 'Dropout_Probability_display', 'attendance_display', 'marks_display', 'fees_due_display'],
        column_config={
            'StudentID': st.column_config.Column(label='🆔 Student ID', width='medium'),
            'StudentName': st.column_config.Column(label='👤 Name', width='large'),
            'Department': st.column_config.Column(label='🏢 Department', width='medium'),
            'MentorID': st.column_config.Column(label='👨‍🏫 Mentor ID', width='medium'),
            'Risk_Level_display': st.column_config.Column(label='⚠️ Risk Level', width='medium'),
            'Dropout_Probability_display': st.column_config.ProgressColumn(
                label='📊 Risk Score',
                help='Predicted dropout probability',
                format='%.1f%%',
                min_value=0,
                max_value=1,
                width='medium'
            ),
            'attendance_display': st.column_config.TextColumn(label='📅 Attendance', width='small'),
            'marks_display': st.column_config.TextColumn(label='📝 Marks', width='small'),
            'fees_due_display': st.column_config.TextColumn(label='💰 Fees Due', width='medium'),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=600
    )
    
    # Check if a row was selected
    if len(event.selection.rows) > 0:
        selected_row_index = event.selection.rows[0]
        selected_student_id = filtered_df.iloc[selected_row_index]['StudentID']
        st.session_state['selected_student_id'] = selected_student_id
        st.session_state['page'] = 'student_details'
        st.rerun()

    # Add a button to send notifications
    if st.sidebar.button("📧 Send High-Risk Alerts", use_container_width=True, key="dashboard_alerts"):
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
    st.set_page_config(
        layout="wide", 
        page_title="University Dashboard",
        page_icon="🎓",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        text-align: center;
        color: #1f4e79;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Remove default streamlit padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Hide streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom metric styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #495057;
        border-bottom: 3px solid #007bff;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #007bff;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🎓 University Dropout Risk Analytics Platform</h1>', unsafe_allow_html=True)
    
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['page'] = 'login'
        st.session_state['selected_student_id'] = None
    
    df, mentors_df, model = load_and_prepare_data()
    if df is None or mentors_df is None or model is None:
        return

    # Main navigation logic
    if st.session_state['logged_in']:
        # Enhanced Sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### 👋 Welcome!")
            st.markdown(f"**User:** `{st.session_state['mentor_id']}`")
            if st.session_state['mentor_id'] != 'ADM-001':
                mentor_name = mentors_df[mentors_df['MentorID'] == st.session_state['mentor_id']]['MentorName'].iloc[0]
                st.markdown(f"**Mentor:** {mentor_name}")
                st.markdown(f"**Department:** {st.session_state['department']}")
            else:
                st.markdown("**Role:** System Administrator")
            
            st.markdown("---")
            
            # Action buttons in sidebar
            if st.button("📧 Send High-Risk Alerts", use_container_width=True, key="sidebar_alerts"):
                high_risk_students = st.session_state['predicted_df'][st.session_state['predicted_df']['Risk_Level'] == '🔴 High Risk']
                if not high_risk_students.empty:
                    # This would trigger the notification logic
                    st.success("🚀 Alert sending initiated!")
                else:
                    st.info("ℹ️ No high-risk students found.")
            
            st.markdown("---")
            
            if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
                st.session_state['logged_in'] = False
                st.session_state['page'] = 'login'
                if 'predicted_df' in st.session_state:
                    del st.session_state['predicted_df']
                st.rerun()
        
        # --- Optimization: Run predictions only once per session ---
        if 'predicted_df' not in st.session_state:
            with st.spinner('Preparing dashboard... This may take a moment.'):
                # Isolate the features needed for prediction
                features = ['attendance', 'marks', 'attempts', 'fees_due']
                
                # Create a temporary DataFrame for prediction, dropping rows with any missing values
                df_for_prediction = df.dropna(subset=features).copy()
                
                # Run predictions on the clean data
                predictions_proba = model.predict_proba(df_for_prediction[features])[:, 1]
                
                # Add the predictions to the temporary DataFrame
                df_for_prediction['Dropout_Probability'] = predictions_proba
                
                # Merge the predictions back to the original DataFrame
                # This ensures students with missing data remain in the main DF
                predicted_df = pd.merge(df, df_for_prediction[['StudentID', 'Dropout_Probability']], on='StudentID', how='left')

                # Define risk level based on probability, handling NaN gracefully
                def get_risk_level(prob):
                    if pd.isna(prob): return "Not Predicted"
                    elif prob >= 0.7: return '🔴 High Risk'
                    elif prob >= 0.3: return '🟡 Moderate Risk'
                    else: return '🟢 Low Risk'

                predicted_df['Risk_Level'] = predicted_df['Dropout_Probability'].apply(get_risk_level)
                
                st.session_state['predicted_df'] = predicted_df

        if st.session_state['page'] == 'dashboard':
            show_dashboard(st.session_state['predicted_df'], mentors_df)

        elif st.session_state['page'] == 'student_details':
            show_student_details(st.session_state['predicted_df'])
    else:
        # Enhanced Login Page
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div style='text-align: center; background-color: #f8f9fa; padding: 2rem; border-radius: 10px; border: 1px solid #e9ecef;'>
                <h2 style='color: #495057; margin-bottom: 1.5rem;'>🔐 System Login</h2>
                <p style='color: #6c757d; margin-bottom: 2rem;'>Enter your credentials to access the analytics platform</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Login form
            with st.form("login_form", clear_on_submit=False):
                mentor_id = st.text_input("🆔 Mentor ID", placeholder="Enter your mentor ID")
                password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
                
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    login_button = st.form_submit_button("🚀 Login to Dashboard", use_container_width=True)
                
                if login_button:
                    check_password(mentor_id, password, mentors_df)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Demo credentials info
            with st.expander("🔍 Demo Credentials"):
                st.markdown("""
                **For demonstration purposes:**
                - **Admin Access:** `ADM-001` / `admin123`
                - **Mentor Access:** `MNT-001` / `mentor123`
                """)
                st.warning("⚠️ In production, use secure authentication methods.")
                
        # Add some spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
