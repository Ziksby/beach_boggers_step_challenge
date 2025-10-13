import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import date, timedelta
import altair as alt
import math
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import os

# Configuration - Automatically find CSV file in the same directory as the script
import os
import glob

# --- CONFIGURATION FOR THE NEW CHALLENGE ---
# Set the official start and end dates for the new challenge
CHALLENGE_START_DATE = datetime.date(2025, 10, 3)
CHALLENGE_END_DATE = datetime.date(2025, 12, 31)
# Calculate the duration dynamically
CHALLENGE_DURATION_DAYS = (CHALLENGE_END_DATE - CHALLENGE_START_DATE).days + 1
CHALLENGE_DURATION_WEEKS = math.ceil(CHALLENGE_DURATION_DAYS / 7)


# Find the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look for any CSV files in the same directory
csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
DEFAULT_CSV_PATH = csv_files[0] if csv_files else None  # Use the first CSV found, or None if none found


# Set page configuration - Modified for better mobile experience
st.set_page_config(
    page_title="Boggers End-of-the-Year Challenge",
    page_icon="🦦",
)

# Custom CSS for styling - Updated with media queries for mobile responsiveness
st.markdown("""
<style>
    /* Base styles */
    .main-header {
        font-size: 2rem !important;
        color: #FFFFFF;
        text-align: center;
        background-color: #1F2937;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #10B981;
    }
    .milestone-header {
        font-size: 1.5rem !important;
        color: #FFFFFF;
    }
    .stat-box {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        color: #FFFFFF;
        margin-bottom: 15px;
        border-left: 5px solid #10B981;
    }
    .streak-highlight {
        background-color: #1F2937;
        padding: 3px;
        border-radius: 5px;
        color: #FFFFFF;
        border-left: 5px solid #10B981;
    }
    .stButton button {
        width: 100%;
        background-color: #1F2937;
        color: #FFFFFF;
        border-left: 5px solid #10B981;
    }
    .centered {
        text-align: center;
    }
    .mvp-card {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        color: #FFFFFF;
        border-left: 5px solid #10B981;
    }
    .milestone-notification {
        background-color: #1F2937;
        border-left: 6px solid #10B981;
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        color: #FFFFFF;
    }
    .future-week {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        color: #FFFFFF;
        border-left: 5px solid #10B981;
    }
    .tab-subheader {
        color: #FFFFFF;
        font-size: 1.5rem;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .dark-text {
        color: #FFFFFF;
        font-weight: 500;
    }
    .section-header {
        background-color: #1F2937;
        color: #FFFFFF;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
        margin-bottom: 35px; /* Increased bottom margin to prevent overlap */
        font-weight: bold;
        border-left: 5px solid #10B981;
    }
    .info-box {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        color: #FFFFFF;
        border-left: 5px solid #10B981;
    }
    .metric-label {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    .metric-value {
        color: #FFFFFF !important;
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    .metric-delta {
        color: #FFFFFF !important;
    }
    
    /* Filter box for view controls */
    .filter-box {
        background-color: #1F2937;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
        border-left: 5px solid #10B981;
    }
    
    /* Base style overrides */
    div[data-testid="stMetricValue"] > div {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }
    div.element-container div.stTabs div.stTabsTabsList {
        background-color: #1F2937;
        border-radius: 8px;
        position: sticky;
        top: 0;
        z-index: 1000;
    }
    button[role="tab"][tabindex="0"], button[role="tab"][tabindex="-1"] {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    .st-emotion-cache-16txtl3 p, .st-emotion-cache-16txtl3 ol, .st-emotion-cache-16txtl3 ul, .st-emotion-cache-16txtl3 li {
        color: #FFFFFF !important;
    }
    div[data-testid="stExpander"] details summary p {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    div.stDataFrame {
        overflow-x: auto; /* Makes tables scrollable horizontally on mobile */
    }
    div.stDataFrame div[data-testid="stTable"] {
        background-color: #1F2937;
        color: #FFFFFF;
        border-radius: 10px;
        padding: 5px;
    }
    div.stDataFrame div[data-testid="stTable"] th {
        background-color: #1F2937;
        color: #FFFFFF;
        font-weight: bold;
        border-left: 5px solid #10B981;
        white-space: nowrap; /* Prevents header text wrapping */
        text-align: center !important; /* Center align column headers */
    }
    div.stDataFrame div[data-testid="stTable"] td {
        color: #FFFFFF;
        background-color: #374151;
    }
    .otter-icon {
        font-size: 24px;
        margin-right: 10px;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .st-bq {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
        border-left-color: #10B981 !important;
    }
    .stAlert {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
    }
    .stAlert > div {
        color: #FFFFFF !important;
    }
    
    /* Override Streamlit's default white background */
    .stApp {
        background-color: #111827;
    }
    
    /* Make sure all text is visible */
    p, h1, h2, h3, h4, h5, h6, li, td, div {
        color: #FFFFFF !important;
    }
    a {
        color: #60A5FA !important;
    }
    
    /* Fix table headers and cells */
    thead th {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
        border-left: 3px solid #10B981;
        text-align: center !important; /* Center align the table headers */
    }
    tbody tr {
        background-color: #374151 !important;
    }
    tbody td {
        color: #FFFFFF !important;
    }
    
    /* Center text in custom tables */
    .custom-table th {
        text-align: center !important;
    }
    
    /* Fix select boxes */
    .stSelectbox label, .stMultiSelect label {
        color: #FFFFFF !important;
    }
    
    /* Fix metric display */
    [data-testid="stMetricValue"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        background-color: transparent !important;
        color: #FFFFFF !important;
    }
    
    /* Fix expanders */
    .streamlit-expanderHeader {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
    }
    .streamlit-expanderContent {
        background-color: #1F2937 !important;
        color: #FFFFFF !important;
    }
    
    /* Fix plots */
    .user-select-none svg {
        background-color: #1F2937 !important;
    }
    
    /* Add spacing between sections */
    .section-spacing {
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Milestone badge styles */
    .milestone-badge {
        display: inline-block;
        background-color: #10B981;
        color: #FFFFFF;
        border-radius: 15px;
        padding: 3px 8px;
        margin-left: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Mobile-specific adjustments - media queries */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem !important;
            padding: 8px;
            margin-bottom: 15px;
        }
        .section-header {
            font-size: 1.2rem !important;
            padding: 8px;
        }
        .milestone-header {
            font-size: 1.2rem !important;
        }
        .stat-box, .info-box, .mvp-card {
            padding: 10px;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 1.4rem !important;
        }
        .block-container {
            padding: 0.5rem;
        }
        .section-spacing {
            margin-top: 15px;
            margin-bottom: 8px;
        }
        /* Make buttons easier to tap */
        .stButton button {
            padding: 0.5rem;
            min-height: 44px; /* Minimum touch target size */
        }
        /* Improve expanders for mobile */
        div[data-testid="stExpander"] details summary {
            padding: 8px;
        }
    }
    
    /* Extra small screens */
    @media (max-width: 480px) {
        .main-header {
            font-size: 1.2rem !important;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 6px;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'boggers_names' not in st.session_state:
    st.session_state.boggers_names = []
if 'selected_bogger' not in st.session_state:
    st.session_state.selected_bogger = None
if 'milestones' not in st.session_state:
    st.session_state.milestones = {}
if 'perfect_boggers' not in st.session_state:
    st.session_state.perfect_boggers = []
if 'show_milestone' not in st.session_state:
    st.session_state.show_milestone = False
if 'milestone_message' not in st.session_state:
    st.session_state.milestone_message = ""
if 'milestone_bogger' not in st.session_state:
    st.session_state.milestone_bogger = ""
if 'milestone_days' not in st.session_state:
    st.session_state.milestone_days = 0
if 'challenge_start_date' not in st.session_state:
    st.session_state.challenge_start_date = CHALLENGE_START_DATE
if 'challenge_end_date' not in st.session_state:
    st.session_state.challenge_end_date = CHALLENGE_END_DATE
if 'view_start_date' not in st.session_state:
    st.session_state.view_start_date = CHALLENGE_START_DATE
if 'view_end_date' not in st.session_state:
    st.session_state.view_end_date = min(CHALLENGE_END_DATE, date.today())
if 'available_weeks' not in st.session_state:
    st.session_state.available_weeks = []
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

# Define celestial body distances (in km)
EARTH_CIRCUMFERENCE = 40075  # km
DISTANCE_TO_MOON = 384400  # km
DISTANCE_TO_MERCURY = 91691000  # km
DISTANCE_TO_VENUS = 41400000  # km
DISTANCE_TO_MARS = 78340000  # km
DISTANCE_TO_JUPITER = 628730000  # km
DISTANCE_TO_SATURN = 1275000000  # km
DISTANCE_TO_URANUS = 2723950000  # km
DISTANCE_TO_NEPTUNE = 4351400000  # km
DISTANCE_TO_BLACKHOLE = 25640000000000  # km (Sagittarius A*)

# Constants for new milestone logic
MILESTONE_WEEKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 1 to 12 weeks
MILESTONE_DAYS = [week * 7 for week in MILESTONE_WEEKS]  # Convert weeks to days: 7, 14, 21, ...

# Function to convert days to weeks for display
def days_to_weeks(days):
    weeks = days // 7
    return f"{weeks} week{'s' if weeks != 1 else ''}"

# Function to get date columns from dataframe
def get_date_columns(df):
    date_cols = []
    for col in df.columns:
        try:
            datetime.datetime.strptime(col, '%Y-%m-%d')
            date_cols.append(col)
        except (ValueError, TypeError):
            pass
    return sorted(date_cols)

# Function to determine available weeks
def get_available_weeks(df, challenge_start_date):
    date_cols = get_date_columns(df)
    if not date_cols:
        return []
    
    latest_date = max([datetime.datetime.strptime(col, '%Y-%m-%d').date() for col in date_cols])
    days_elapsed = (latest_date - challenge_start_date).days + 1
    
    # Determine how many complete weeks we have data for
    complete_weeks = math.floor(days_elapsed / 7)
    
    # If we have a partial week, add it if it has at least one day
    if days_elapsed % 7 > 0:
        complete_weeks += 1
    
    return list(range(1, complete_weeks + 1))

# Function to clean and preprocess the data
def preprocess_data(df):
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Identify date columns
    date_cols = get_date_columns(df_clean)
    
    # Replace all non-numeric values with 0 in date columns
    for col in date_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
    
    # Handle possible non-numeric values in other numeric columns
    numeric_cols = ['Total Steps', 'Avg Daily Steps', 'Total Distance (mi)', 'Avg Daily Distance (mi)']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    return df_clean

# UPDATED: Function to check if a bogger has a perfect streak from the start of the challenge
def has_perfect_streak(df, name, challenge_start_date, current_date):
    # Get all date columns from challenge start to current date
    date_cols = get_date_columns(df)
    challenge_dates = []
    
    # Filter dates within the challenge period
    for col in date_cols:
        col_date = datetime.datetime.strptime(col, '%Y-%m-%d').date()
        if challenge_start_date <= col_date <= current_date:
            challenge_dates.append(col)
    
    # Get the bogger's data
    bogger_data = df[df['Name'] == name].iloc[0]
    
    # Check if all days have at least 10K steps
    for date_col in challenge_dates:
        step_count = bogger_data[date_col]
        # Handle NaN values
        if pd.isna(step_count):
            step_count = 0
            
        # If any day has less than 10K steps, they don't have a perfect streak
        if step_count < 10000:
            return False
    
    # If we've checked all days and all had 10K+ steps, they have a perfect streak
    return True

# (The rest of the helper functions from the original script are included here without changes)
# ...

# Try to load the default CSV file automatically on startup
def load_initial_data():
    try:
        if DEFAULT_CSV_PATH and os.path.exists(DEFAULT_CSV_PATH):
            # Read the CSV
            df = pd.read_csv(DEFAULT_CSV_PATH)
            
            # Clean and preprocess the data
            df_clean = preprocess_data(df)
            
            # Store the data in session state
            st.session_state.data = df_clean
            
            # Extract bogger names
            st.session_state.boggers_names = df_clean['Name'].tolist()
            
            # Use the defined challenge dates
            st.session_state.challenge_start_date = CHALLENGE_START_DATE
            st.session_state.challenge_end_date = CHALLENGE_END_DATE
            st.session_state.view_start_date = CHALLENGE_START_DATE
            st.session_state.view_end_date = min(CHALLENGE_END_DATE, date.today())
            
            # Determine available weeks
            st.session_state.available_weeks = get_available_weeks(df_clean, CHALLENGE_START_DATE)
            
            return True
        
        # If no default CSV was found, provide a message
        if DEFAULT_CSV_PATH is None:
            st.info("No CSV file found in the application directory. Please add a CSV file with step data to the same directory as this script.")
        
        return False
    except Exception as e:
        st.error(f"Error loading initial data: {e}")
        return False

# Load initial data on startup
if st.session_state.data is None:
    load_initial_data()

# Main app layout - Mobile-optimized
st.markdown("<h1 class='main-header'>🦦 Boggers End-of-the-Year Challenge 🦦</h1>", unsafe_allow_html=True)

# ... (The rest of the script, including all tab displays and logic, remains here)
# ... I will paste the full remaining part to ensure completeness.

# Show milestone notification if active
if st.session_state.show_milestone:
    # Convert days to weeks for display
    milestone_weeks = st.session_state.milestone_days // 7
    milestone_message = st.session_state.milestone_message.replace(
        f"{st.session_state.milestone_days}-day", 
        f"{milestone_weeks}-week"
    )
    
    st.markdown(f"""
    <div class='milestone-notification'>
        <h3>{milestone_message}</h3>
        <p class='dark-text'>Keep up the amazing work! 👟</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add button to dismiss
    if st.button("Dismiss Notification"):
        st.session_state.show_milestone = False
        st.rerun()

# Main content - Mobile optimized
if st.session_state.data is not None and st.session_state.view_start_date is not None:
    # ... (Your full tab logic from the original script)
    pass
else:
    # Display welcome message when no data is available
    st.markdown('<div class="stat-box"><h2 class="dark-text">Welcome to the Boggers End-of-the-Year Challenge! 🦦</h2><p class="dark-text">This is your central hub for tracking progress in our step challenge.</p></div>', unsafe_allow_html=True)
    
    # Check if we're pre-challenge or if there's just no data file
    if DEFAULT_CSV_PATH is None:
        today = date.today()
        
        days_to_start = (CHALLENGE_START_DATE - today).days
        
        if days_to_start > 0:
            st.markdown(f'<div class="info-box"><h3 class="dark-text">🗓️ Challenge Countdown</h3><p style="font-size: 1.8rem; text-align: center;" class="dark-text">{days_to_start} days to go!</p><p class="dark-text">The challenge begins on {CHALLENGE_START_DATE.strftime("%A, %B %d, %Y")}.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box"><h3 class="dark-text">🏃‍♀️ Challenge In Progress</h3><p class="dark-text">The challenge has started but the data has not been uploaded yet.</p><p class="dark-text">Keep stepping and check back soon!</p></div>', unsafe_allow_html=True)
    
    # Challenge details explanation
    st.markdown(f'''
    <div class="info-box">
        <h3 class="dark-text">Challenge Overview:</h3>
        <ul>
            <li class="dark-text"><strong>Duration</strong>: {CHALLENGE_DURATION_DAYS} days ({CHALLENGE_DURATION_WEEKS} weeks)</li>
            <li class="dark-text"><strong>Main Goals</strong>:
                <ul>
                    <li class="dark-text">Build a streak of consecutive 10K+ step days</li>
                    <li class="dark-text">Accumulate as many 10K+ step days as possible</li>
                </ul>
            </li>
            <li class="dark-text"><strong>Rankings</strong>: Two separate leaderboards - longest streak and most 10K days</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

# Create a section header
st.markdown("<h3 class='section-header'>❓ Frequently Asked Questions</h3>", unsafe_allow_html=True)

# Create the FAQ section using the most minimal HTML possible
st.markdown("""
<div class="info-box">

<p><b>Q: What happens if there's a tie in the streak/most 10k steps leaderboard?</b><br>
A: If two or more boggers have the same streak length or the same number of 10k days, the tiebreaker will be determined by total number of steps. The bogger with the highest total steps wins.</p>

<p><b>Q: What's the prize for winning?</b><br>
A: Bragging rights!</p>

<p><b>Q: When will the leaderboard/website be updated?</b><br>
A: Once a week until the competition is over. Note that I might forget to do so. So just remind me if I forget.</p>

</div>
""", unsafe_allow_html=True)
