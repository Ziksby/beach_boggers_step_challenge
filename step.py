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

# Find the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Look for any CSV files in the same directory
csv_files = glob.glob(os.path.join(script_dir, "*.csv"))
DEFAULT_CSV_PATH = csv_files[0] if csv_files else None  # Use the first CSV found, or None if none found

# Set page configuration - Modified for better mobile experience
st.set_page_config(
    page_title="84-Day Step Challenge",
    page_icon="🦦",
    # Remove layout="wide" to allow responsive behavior
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed for mobile
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
    .gold-name {
        font-weight: bold;
        color: #FFD700;
        text-shadow: 0px 0px 5px #FFA500;
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
        margin-bottom: 15px;
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
    }
    tbody tr {
        background-color: #374151 !important;
    }
    tbody td {
        color: #FFFFFF !important;
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

# Function to detect if we're on a mobile device (based on window width)
# This will be used to adjust layouts dynamically
st.markdown("""
<script>
const isMobile = () => {
    return window.innerWidth <= 768;
};

// Store device type in sessionStorage
sessionStorage.setItem('isMobile', isMobile());
</script>
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
if 'show_milestone' not in st.session_state:
    st.session_state.show_milestone = False
if 'milestone_message' not in st.session_state:
    st.session_state.milestone_message = ""
if 'milestone_bogger' not in st.session_state:
    st.session_state.milestone_bogger = ""
if 'milestone_days' not in st.session_state:
    st.session_state.milestone_days = 0
if 'challenge_start_date' not in st.session_state:
    st.session_state.challenge_start_date = None
if 'challenge_end_date' not in st.session_state:
    st.session_state.challenge_end_date = None
if 'view_start_date' not in st.session_state:
    st.session_state.view_start_date = None
if 'view_end_date' not in st.session_state:
    st.session_state.view_end_date = None
if 'available_weeks' not in st.session_state:
    st.session_state.available_weeks = []
if 'is_mobile' not in st.session_state:
    st.session_state.is_mobile = False  # Will be updated based on screen size

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

# All the existing functions remain unchanged
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

# Function to detect challenge dates from CSV
def detect_challenge_dates(df):
    date_cols = get_date_columns(df)
    if not date_cols:
        # Default to today if no dates found
        today = date.today()
        return today, today + timedelta(days=83)
    
    # Get earliest and latest date
    earliest_date_str = min(date_cols)
    latest_date_str = max(date_cols)
    
    start_date = datetime.datetime.strptime(earliest_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(latest_date_str, '%Y-%m-%d').date()
    
    # If we have less than 84 days of data, end date will be start date + 83 days
    if (end_date - start_date).days < 83:
        end_date = start_date + timedelta(days=83)
    
    return start_date, end_date

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

# Function to calculate streaks and metrics
def calculate_metrics(df, date_cols, start_date_str, end_date_str):
    # Convert date strings to datetime
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    # Filter date columns within the selected range
    filtered_date_cols = [col for col in date_cols if start_date <= datetime.datetime.strptime(col, '%Y-%m-%d').date() <= end_date]
    
    # Initialize results dataframe
    results = pd.DataFrame()
    results['Name'] = df['Name']
    
    # Calculate 10K step days
    results['days_with_10k'] = 0
    results['current_streak'] = 0
    results['longest_streak'] = 0
    results['total_steps'] = 0
    results['total_distance_km'] = 0
    
    # For milestone tracking
    prev_longest_streaks = {}
    
    # Process each bogger
    for idx, row in df.iterrows():
        days_with_10k = 0
        current_streak = 0
        longest_streak = 0
        max_streak_start = None
        current_streak_start = None
        total_steps = 0
        total_distance_km = 0
        
        # Get bogger name
        name = row['Name']
        
        # Initialize previous longest streak if not already tracked
        if name not in prev_longest_streaks:
            prev_longest_streaks[name] = 0
            
        # Get their previous longest streak
        prev_longest_streak = prev_longest_streaks[name]
        
        # Iterate through dates in range
        for i, date_col in enumerate(filtered_date_cols):
            step_count = row[date_col]
            
            # Check if step count is NaN or NA, convert to 0
            if pd.isna(step_count):
                step_count = 0
            
            total_steps += step_count
            
            # Get distance if available
            if 'Total Distance (mi)' in df.columns:
                try:
                    # Convert miles to kilometers
                    total_distance_mi = float(row['Total Distance (mi)'])
                    if pd.isna(total_distance_mi):
                        total_distance_mi = 0
                    total_distance_km = total_distance_mi * 1.60934
                except (ValueError, TypeError):
                    total_distance_km = 0
            
            # Check if step count meets or exceeds 10K
            if step_count >= 10000:
                days_with_10k += 1
                current_streak += 1
                
                # Keep track of current streak start
                if current_streak == 1:
                    current_streak_start = date_col
                
                # Update longest streak if current streak is longer
                if current_streak > longest_streak:
                    longest_streak = current_streak
                    max_streak_start = current_streak_start
            else:
                # Reset current streak
                current_streak = 0
                current_streak_start = None
        
        # Store the metrics
        results.loc[idx, 'days_with_10k'] = int(days_with_10k)
        results.loc[idx, 'current_streak'] = int(current_streak)
        results.loc[idx, 'longest_streak'] = int(longest_streak)
        results.loc[idx, 'total_steps'] = int(total_steps)
        results.loc[idx, 'total_distance_km'] = total_distance_km
        
        # Check for new streak milestones
        if int(longest_streak) > int(prev_longest_streak):
            # Check for milestone crossings (10, 20, 30, etc.)
            for milestone in [10, 20, 30, 40, 50, 60, 70, 80]:
                if int(prev_longest_streak) < milestone <= int(longest_streak):
                    # Record new milestone
                    if name not in st.session_state.milestones:
                        st.session_state.milestones[name] = []
                    
                    # Add milestone if not already there
                    if milestone not in st.session_state.milestones[name]:
                        st.session_state.milestones[name].append(milestone)
                        
                        # Set milestone notification
                        st.session_state.show_milestone = True
                        st.session_state.milestone_message = f"🎉 Amazing achievement! {name} has reached a {milestone}-day streak!"
                        st.session_state.milestone_bogger = name
                        st.session_state.milestone_days = milestone
        
        # Update previous longest streak
        prev_longest_streaks[name] = int(longest_streak)
    
    # Calculate rankings by longest streak
    results['streak_rank'] = (-results['longest_streak']).rank(method='min').astype(int)
    
    # Calculate rankings by most 10K days
    results['days_rank'] = (-results['days_with_10k']).rank(method='min').astype(int)
    
    # Check for perfect streak potential
    if st.session_state.challenge_start_date:
        challenge_start = st.session_state.challenge_start_date
        today = date.today()
        days_since_start = (today - challenge_start).days + 1
        
        # Only apply gold highlight if we're within the challenge period
        if today >= challenge_start and days_since_start > 0:
            results['perfect_streak_potential'] = results['current_streak'] >= days_since_start
        else:
            results['perfect_streak_potential'] = False
    else:
        results['perfect_streak_potential'] = False
    
    return results

# Function to create what-if scenarios for streak challenge
def create_streak_scenario(selected_name, metrics_df):
    # Get the selected bogger's row
    selected_row = metrics_df[metrics_df['Name'] == selected_name].iloc[0]
    selected_rank = selected_row['streak_rank']
    
    # If rank is 1, they're already at the top
    if selected_rank == 1:
        return "🏆 You're at the top! Keep doing you!"
    
    # Find the person directly above in STREAK rank
    above_rank = selected_rank - 1
    above_persons = metrics_df[metrics_df['streak_rank'] == above_rank]
    
    # If multiple people have the same streak rank, find the one with the most steps
    if len(above_persons) > 1:
        above_person = above_persons.sort_values('total_steps', ascending=False).iloc[0]
    else:
        above_person = above_persons.iloc[0]
    
    # Calculate the streak gap
    streak_gap = above_person['longest_streak'] - selected_row['longest_streak']
    
    # In case of same streak but different steps (tiebreaker)
    if streak_gap == 0:
        steps_gap = above_person['total_steps'] - selected_row['total_steps']
        if steps_gap > 0:
            return f"👟 You have the same streak as {above_person['Name']} but need {steps_gap+1:,} more total steps to overtake them in the tiebreaker."
        else:
            # This shouldn't happen normally, but handle the case anyway
            return f"🤔 There seems to be a tie with {above_person['Name']} on streak and total steps. Check with the admin."
    
    # Calculate days left in challenge
    today = date.today()
    if st.session_state.challenge_end_date:
        days_left = (st.session_state.challenge_end_date - today).days
        days_left = max(0, days_left)  # Ensure it doesn't go negative
    else:
        days_left = 84  # Default to full challenge length if no end date set
    
    # Check if it's impossible to catch up based on days left
    if days_left < streak_gap:
        return f"This is the best you can do for now. There aren't enough days left to catch up to the next rank."
    
    # Generate streak advice
    if selected_row['current_streak'] == 0:
        return f"🔥 Start a new streak and maintain it for {streak_gap+1} days to overtake {above_person['Name']} provided they do not improve."
    else:
        return f"🔥 Keep your current streak going for {streak_gap+1} more days to overtake {above_person['Name']} provided they do not improve."

# Function to create what-if scenarios for 10K days challenge
def create_10k_days_scenario(selected_name, metrics_df):
    # Get the selected bogger's row
    selected_row = metrics_df[metrics_df['Name'] == selected_name].iloc[0]
    selected_rank = selected_row['days_rank']
    
    # If rank is 1, they're already at the top
    if selected_rank == 1:
        return "🏆 You're at the top! Keep doing you!"
    
    # Find the person directly above in DAYS rank
    above_rank = selected_rank - 1
    above_persons = metrics_df[metrics_df['days_rank'] == above_rank]
    
    # If multiple people have the same days rank, find the one with the most steps
    if len(above_persons) > 1:
        above_person = above_persons.sort_values('total_steps', ascending=False).iloc[0]
    else:
        above_person = above_persons.iloc[0]
    
    # Calculate the 10K days gap
    days_10k_gap = above_person['days_with_10k'] - selected_row['days_with_10k']
    
    # In case of same 10K days but different steps (tiebreaker)
    if days_10k_gap == 0:
        steps_gap = above_person['total_steps'] - selected_row['total_steps']
        if steps_gap > 0:
            return f"👟 You have the same number of 10K days as {above_person['Name']} but need {steps_gap+1:,} more total steps to overtake them in the tiebreaker."
        else:
            # This shouldn't happen normally, but handle the case anyway
            return f"🤔 There seems to be a tie with {above_person['Name']} on 10K days and total steps. Check with the admin."
    
    # Calculate days left in challenge
    today = date.today()
    if st.session_state.challenge_end_date:
        days_left = (st.session_state.challenge_end_date - today).days
        days_left = max(0, days_left)  # Ensure it doesn't go negative
    else:
        days_left = 84  # Default to full challenge length if no end date set
    
    # Check if it's impossible to catch up based on days left
    if days_left < days_10k_gap:
        return f"This is the best you can do for now. There aren't enough days left to catch up to the next rank."
    
    # Generate 10K days advice
    return f"👟 You need {days_10k_gap+1} more 10K-step days to overtake {above_person['Name']} provided they do not improve."

# Function to find best week for 10K days
def find_best_week_10k(df, date_cols, bogger_name):
    # Get the bogger's data
    bogger_data = df[df['Name'] == bogger_name].iloc[0]
    
    # Initialize variables to track best week
    best_week_start = None
    best_week_steps = 0
    best_week_10k_days = 0
    
    # Loop through weeks (using 7-day windows)
    for i in range(0, len(date_cols) - 6):
        week_cols = date_cols[i:i+7]
        week_start = week_cols[0]
        
        # Calculate week metrics
        week_steps = 0
        week_10k_days = 0
        
        for day in week_cols:
            step_count = bogger_data[day]
            # Handle NaN values
            if pd.isna(step_count):
                step_count = 0
                
            week_steps += step_count
            if step_count >= 10000:
                week_10k_days += 1
        
        # Check if this is the best week - prioritize 10K days count
        if week_10k_days > best_week_10k_days or (week_10k_days == best_week_10k_days and week_steps > best_week_steps):
            best_week_start = week_start
            best_week_steps = week_steps
            best_week_10k_days = week_10k_days
    
    # If we found a best week
    if best_week_start:
        best_week_start_date = datetime.datetime.strptime(best_week_start, '%Y-%m-%d').date()
        best_week_end_date = best_week_start_date + timedelta(days=6)
        
        return {
            'start_date': best_week_start_date.strftime('%Y-%m-%d'),
            'end_date': best_week_end_date.strftime('%Y-%m-%d'),
            'total_steps': best_week_steps,
            'days_with_10k': best_week_10k_days,
            'avg_steps': best_week_steps / 7
        }
    
    return None

# Function to find best week for streak
def find_best_week_streak(df, date_cols, bogger_name):
    # Get the bogger's data
    bogger_data = df[df['Name'] == bogger_name].iloc[0]
    
    # Initialize variables to track best week
    best_week_start = None
    best_week_steps = 0
    best_week_longest_streak = 0
    
    # Loop through weeks (using 7-day windows)
    for i in range(0, len(date_cols) - 6):
        week_cols = date_cols[i:i+7]
        week_start = week_cols[0]
        
        # Calculate week metrics
        week_steps = 0
        current_streak = 0
        longest_streak = 0
        
        for day in week_cols:
            step_count = bogger_data[day]
            # Handle NaN values
            if pd.isna(step_count):
                step_count = 0
                
            week_steps += step_count
            
            if step_count >= 10000:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 0
        
        # Check if this is the best week - prioritize longest streak
        if longest_streak > best_week_longest_streak or (longest_streak == best_week_longest_streak and week_steps > best_week_steps):
            best_week_start = week_start
            best_week_steps = week_steps
            best_week_longest_streak = longest_streak
    
    # If we found a best week
    if best_week_start:
        best_week_start_date = datetime.datetime.strptime(best_week_start, '%Y-%m-%d').date()
        best_week_end_date = best_week_start_date + timedelta(days=6)
        
        return {
            'start_date': best_week_start_date.strftime('%Y-%m-%d'),
            'end_date': best_week_end_date.strftime('%Y-%m-%d'),
            'total_steps': best_week_steps,
            'longest_streak': best_week_longest_streak,
            'avg_steps': best_week_steps / 7
        }
    
    return None

# Function to calculate weekly MVPs
def calculate_weekly_mvps(df, date_cols):
    # Group date columns by week
    weeks = []
    current_week = []
    
    for i, col in enumerate(date_cols):
        current_week.append(col)
        # If we have 7 days or we're at the end of the date columns
        if len(current_week) == 7 or i == len(date_cols) - 1:
            weeks.append(current_week)
            current_week = []
    
    # Calculate MVPs for each week
    weekly_mvps = []
    
    for i, week_cols in enumerate(weeks):
        week_start = datetime.datetime.strptime(week_cols[0], '%Y-%m-%d').date()
        week_end = (week_start + timedelta(days=len(week_cols) - 1))
        
        # Check if this week is in the future
        if week_end > date.today():
            weekly_mvps.append({
                'week_num': i + 1,
                'start_date': week_start.strftime('%Y-%m-%d'),
                'end_date': week_end.strftime('%Y-%m-%d'),
                'future_week': True
            })
            continue
        
        # Initialize weekly metrics
        week_metrics = pd.DataFrame()
        week_metrics['Name'] = df['Name']
        week_metrics['days_with_10k'] = 0
        week_metrics['longest_streak'] = 0
        week_metrics['total_steps'] = 0
        
        # Calculate metrics for this week
        for idx, row in df.iterrows():
            days_with_10k = 0
            current_streak = 0
            longest_streak = 0
            total_steps = 0
            
            for day in week_cols:
                step_count = row[day]
                # Handle NaN values
                if pd.isna(step_count):
                    step_count = 0
                
                total_steps += step_count
                
                if step_count >= 10000:
                    days_with_10k += 1
                    current_streak += 1
                    longest_streak = max(longest_streak, current_streak)
                else:
                    current_streak = 0
            
            # Store the metrics
            week_metrics.loc[idx, 'days_with_10k'] = days_with_10k
            week_metrics.loc[idx, 'longest_streak'] = longest_streak
            week_metrics.loc[idx, 'total_steps'] = total_steps
        
        # Find streak MVP(s) with proper tiebreaker based on total steps
        max_streak = week_metrics['longest_streak'].max()
        streak_tied_boggers = week_metrics[week_metrics['longest_streak'] == max_streak]
        
        if len(streak_tied_boggers) > 1:
            # Check if there's a tiebreaker based on total steps
            max_steps_among_tied = streak_tied_boggers['total_steps'].max()
            complete_tie_boggers = streak_tied_boggers[streak_tied_boggers['total_steps'] == max_steps_among_tied]
            
            if len(complete_tie_boggers) > 1:
                # Complete tie - multiple MVPs
                streak_mvp = {
                    'name': ", ".join(complete_tie_boggers['Name'].tolist()),
                    'streak': max_streak,
                    'is_tie': True
                }
            else:
                # Tie resolved by steps
                winner = streak_tied_boggers.loc[streak_tied_boggers['total_steps'].idxmax()]
                streak_mvp = {
                    'name': winner['Name'],
                    'streak': max_streak,
                    'is_tie': False
                }
        else:
            # No tie
            streak_mvp = {
                'name': streak_tied_boggers.iloc[0]['Name'],
                'streak': max_streak,
                'is_tie': False
            }
        
        # Find days MVP(s) with proper tiebreaker based on total steps
        max_days = week_metrics['days_with_10k'].max()
        days_tied_boggers = week_metrics[week_metrics['days_with_10k'] == max_days]
        
        if len(days_tied_boggers) > 1:
            # Check if there's a tiebreaker based on total steps
            max_steps_among_tied = days_tied_boggers['total_steps'].max()
            complete_tie_boggers = days_tied_boggers[days_tied_boggers['total_steps'] == max_steps_among_tied]
            
            if len(complete_tie_boggers) > 1:
                # Complete tie - multiple MVPs
                days_mvp = {
                    'name': ", ".join(complete_tie_boggers['Name'].tolist()),
                    'days': max_days,
                    'is_tie': True
                }
            else:
                # Tie resolved by steps
                winner = days_tied_boggers.loc[days_tied_boggers['total_steps'].idxmax()]
                days_mvp = {
                    'name': winner['Name'],
                    'days': max_days,
                    'is_tie': False
                }
        else:
            # No tie
            days_mvp = {
                'name': days_tied_boggers.iloc[0]['Name'],
                'days': max_days,
                'is_tie': False
            }
        
        # Add to weekly MVPs list
        weekly_mvps.append({
            'week_num': i + 1,
            'start_date': week_start.strftime('%Y-%m-%d'),
            'end_date': week_end.strftime('%Y-%m-%d'),
            'streak_mvp': streak_mvp,
            'days_mvp': days_mvp,
            'future_week': False
        })
    
    return weekly_mvps

# Function to format distance milestones
def format_distance_milestone(total_distance_km):
    if total_distance_km < EARTH_CIRCUMFERENCE:
        # Less than Earth's circumference
        percentage = (total_distance_km / EARTH_CIRCUMFERENCE) * 100
        return f"{percentage:.1f}% of the way around the Earth", percentage
    
    # Calculations for other celestial bodies
    bodies = [
        ("the Moon", DISTANCE_TO_MOON),
        ("Mercury", DISTANCE_TO_MERCURY),
        ("Venus", DISTANCE_TO_VENUS),
        ("Mars", DISTANCE_TO_MARS),
        ("Jupiter", DISTANCE_TO_JUPITER),
        ("Saturn", DISTANCE_TO_SATURN),
        ("Uranus", DISTANCE_TO_URANUS),
        ("Neptune", DISTANCE_TO_NEPTUNE),
        ("Sagittarius A* (our galaxy's black hole)", DISTANCE_TO_BLACKHOLE)
    ]
    
    for i, (body, distance) in enumerate(bodies):
        if total_distance_km < distance:
            percentage = (total_distance_km / distance) * 100
            return f"{percentage:.2f}% of the way to {body}", percentage
    
    # If we've exceeded all distances
    percentage = (total_distance_km / DISTANCE_TO_BLACKHOLE) * 100
    return f"{percentage:.4f}% of the way to the galactic center", percentage

# Function to generate community stats
def generate_community_stats(df, date_cols, end_date):
    # Calculate total steps across all boggers
    total_steps = 0
    
    # Calculate total distance using the Total Distance column if available
    total_distance_km = 0
    
    if 'Total Distance (mi)' in df.columns:
        for _, row in df.iterrows():
            try:
                # Convert miles to kilometers
                distance_mi = float(row['Total Distance (mi)'])
                if pd.isna(distance_mi):
                    distance_mi = 0
                total_distance_km += distance_mi * 1.60934
            except (ValueError, TypeError):
                pass
    
    # If no distance column or it's invalid, calculate from steps
    if total_distance_km == 0:
        for _, row in df.iterrows():
            for col in date_cols:
                date_val = datetime.datetime.strptime(col, '%Y-%m-%d').date()
                end_date_val = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
                
                if date_val <= end_date_val:
                    step_count = row[col]
                    # Handle NaN values
                    if pd.isna(step_count):
                        step_count = 0
                        
                    total_steps += step_count
        
        # Estimate distance (2000 steps ≈ 1 mile ≈ 1.60934 km)
        total_distance_km = (total_steps / 2000) * 1.60934
    
    # Format for cosmic milestones
    cosmic_milestone, milestone_percentage = format_distance_milestone(total_distance_km)
    
    # Calculate equivalent marathons (42.195 km per marathon)
    marathons = total_distance_km / 42.195
    
    # Calculate today's boggers who hit 10k
    today_col = end_date
    today_boggers = 0
    
    if today_col in df.columns:
        for _, row in df.iterrows():
            step_count = row[today_col]
            # Handle NaN values
            if pd.isna(step_count):
                step_count = 0
                
            if step_count >= 10000:
                today_boggers += 1
    
    return {
        'total_steps': total_steps,
        'total_distance_km': total_distance_km,
        'marathons': marathons,
        'cosmic_milestone': cosmic_milestone,
        'milestone_percentage': milestone_percentage,
        'today_boggers': today_boggers,
        'total_boggers': len(df)
    }

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
            
            # Detect challenge dates from CSV
            if st.session_state.challenge_start_date is None:
                start_date, end_date = detect_challenge_dates(df_clean)
                st.session_state.challenge_start_date = start_date
                st.session_state.challenge_end_date = end_date
                st.session_state.view_start_date = start_date
                st.session_state.view_end_date = min(end_date, date.today())
                
                # Determine available weeks
                st.session_state.available_weeks = get_available_weeks(df_clean, start_date)
            
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
st.markdown("<h1 class='main-header'>🦦 84-Day Boggers Step Challenge 🦦</h1>", unsafe_allow_html=True)

# Show milestone notification if active
if st.session_state.show_milestone:
    st.markdown(f"""
    <div class='milestone-notification'>
        <h3>{st.session_state.milestone_message}</h3>
        <p class='dark-text'>Keep up the amazing work! 👟</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add button to dismiss
    if st.button("Dismiss Notification"):
        st.session_state.show_milestone = False
        st.experimental_rerun()

# Sidebar with filtering options (no file upload)
with st.sidebar:
    st.markdown("<h3 class='section-header'>🦦 Challenge Info 🦦</h3>", unsafe_allow_html=True)
    
    if st.session_state.challenge_start_date:
        # Calculate days left in the challenge
        today = date.today()
        days_left = (st.session_state.challenge_end_date - today).days
        days_left = max(0, days_left)  # Ensure it doesn't go negative
        
        st.markdown(f"""
        <div class='stat-box'>
            <p class='dark-text'><strong>Start Date:</strong> {st.session_state.challenge_start_date.strftime('%Y-%m-%d')}</p>
            <p class='dark-text'><strong>End Date:</strong> {st.session_state.challenge_end_date.strftime('%Y-%m-%d')}</p>
            <p class='dark-text'><strong>Total Days:</strong> 84</p>
            <p class='dark-text'><strong>Days Left:</strong> {days_left}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 class='section-header'>View Filters</h3>", unsafe_allow_html=True)
        
        # Weekly filter option
        # Only show weeks that have data
        available_weeks = st.session_state.available_weeks
        
        # Create week options
        week_options = ["Full Challenge"]
        if available_weeks:
            week_options += [f"Week {i}" for i in available_weeks]
        
        selected_week = st.selectbox("Select Week", week_options)
        
        # Apply week filter if selected
        if selected_week != "Full Challenge":
            week_num = int(selected_week.split(" ")[1])
            week_start = st.session_state.challenge_start_date + timedelta(days=(week_num-1)*7)
            week_end = min(week_start + timedelta(days=6), st.session_state.challenge_end_date)
            
            # Update date selection
            st.session_state.view_start_date = week_start
            st.session_state.view_end_date = min(week_end, date.today())  # Don't show future dates beyond today
            
            # Add a note if this week is in the future
            if week_start > date.today():
                st.warning(f"Week {week_num} hasn't started yet. Data will be shown when available.")
        else:
            # Reset to full available range if "Full Challenge" is selected
            st.session_state.view_start_date = st.session_state.challenge_start_date
            st.session_state.view_end_date = min(st.session_state.challenge_end_date, date.today())
    
    # Display milestone hall of fame
    st.markdown("---")
    st.markdown("<h3 class='section-header'>🏆 Milestone Hall of Fame</h3>", unsafe_allow_html=True)
    
    if st.session_state.milestones:
        for name, milestones in st.session_state.milestones.items():
            st.markdown(f"""
            <div class='stat-box'>
                <p class='dark-text'><strong>{name}</strong></p>
                <p class='dark-text'>{', '.join([f"{m} days" for m in sorted(milestones)])}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No milestones achieved yet! Keep stepping! 🦦")

# Main content - Mobile optimized
if st.session_state.data is not None and st.session_state.view_start_date is not None:
    # Get the data
    df = st.session_state.data
    
    # Get date columns
    date_cols = get_date_columns(df)
    
    # Get view date range as strings
    view_start_date_str = st.session_state.view_start_date.strftime('%Y-%m-%d')
    view_end_date_str = st.session_state.view_end_date.strftime('%Y-%m-%d')
    
    # Filter date columns to only include those that exist in the dataframe
    available_start_date_str = max(min(date_cols), view_start_date_str) if date_cols else view_start_date_str
    available_end_date_str = min(view_end_date_str, max(date_cols)) if date_cols else view_end_date_str
    
    # Calculate metrics
    metrics_df = calculate_metrics(df, date_cols, available_start_date_str, available_end_date_str)
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Leaderboards", "🔍 Details", "📅 MVPs", "📈 Stats"])
    
    with tab1:
        st.markdown("<h2 class='section-header'>Step Challenge Leaderboards</h2>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p class='dark-text'><strong>Viewing data from:</strong> {available_start_date_str} to {available_end_date_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mobile-optimized leaderboards - Stack them vertically instead of side by side
        st.markdown("<h3 class='section-header'>Longest Streak Leaderboard</h3>", unsafe_allow_html=True)
        
        # Create the streak leaderboard
        streak_leaderboard = metrics_df[['streak_rank', 'Name', 'longest_streak', 'perfect_streak_potential']].copy()
        streak_leaderboard = streak_leaderboard.sort_values('streak_rank')
        
        # Highlight names with gold for those with perfect streak potential
        def highlight_gold(name, has_potential):
            if has_potential:
                return f"<span class='gold-name'>{name}</span>"
            return name
        
        # Apply gold highlighting
        streak_leaderboard['Name'] = streak_leaderboard.apply(
            lambda row: highlight_gold(row['Name'], row['perfect_streak_potential']), 
            axis=1
        )
        
        # Format the leaderboard
        streak_leaderboard = streak_leaderboard.rename(columns={
            'streak_rank': 'Rank',
            'longest_streak': 'Longest Streak'
        })
        
        # Drop the perfect_streak_potential column as it's just for styling
        streak_leaderboard = streak_leaderboard.drop('perfect_streak_potential', axis=1)
        
        # Display the streak leaderboard
        st.dataframe(streak_leaderboard, hide_index=True)
        
        st.markdown("<h3 class='section-header'>Most 10K Days Leaderboard</h3>", unsafe_allow_html=True)
        
        # Create the 10K days leaderboard
        days_leaderboard = metrics_df[['days_rank', 'Name', 'days_with_10k', 'perfect_streak_potential']].copy()
        days_leaderboard = days_leaderboard.sort_values('days_rank')
        
        # Apply gold highlighting
        days_leaderboard['Name'] = days_leaderboard.apply(
            lambda row: highlight_gold(row['Name'], row['perfect_streak_potential']), 
            axis=1
        )
        
        # Format the leaderboard
        days_leaderboard = days_leaderboard.rename(columns={
            'days_rank': 'Rank',
            'days_with_10k': '10K Days'
        })
        
        # Drop the perfect_streak_potential column as it's just for styling
        days_leaderboard = days_leaderboard.drop('perfect_streak_potential', axis=1)
        
        # Display the 10K days leaderboard
        st.dataframe(days_leaderboard, hide_index=True)
        
        # Add note about gold names
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>👑 Names in gold indicate boggers who have maintained a perfect streak since the challenge started!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2 class='section-header'>Bogger Details</h2>", unsafe_allow_html=True)
        
        # Dropdown to select bogger
        selected_bogger = st.selectbox(
            "Select a bogger",
            st.session_state.boggers_names,
            index=0 if st.session_state.selected_bogger is None else 
                   st.session_state.boggers_names.index(st.session_state.selected_bogger)
        )
        
        # Store selection in session state
        st.session_state.selected_bogger = selected_bogger
        
        if selected_bogger:
            # Get bogger data
            bogger_data = df[df['Name'] == selected_bogger].iloc[0]
            bogger_metrics = metrics_df[metrics_df['Name'] == selected_bogger].iloc[0]
            
            st.markdown("""
            <div class='section-header'>
                <h3 class='dark-text'>Current Rankings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show rankings in a more mobile-friendly layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Streak Rank", f"#{int(bogger_metrics['streak_rank'])}")
                st.metric("Longest Streak", f"{int(bogger_metrics['longest_streak'])} days")
            
            with col2:
                st.metric("10K Days Rank", f"#{int(bogger_metrics['days_rank'])}")
                st.metric("10K Days Count", f"{int(bogger_metrics['days_with_10k'])} days")
            
            # What-if scenarios
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-header'>What-If Scenarios</h3>", unsafe_allow_html=True)
            
            # Display scenarios in a mobile-friendly vertical layout
            st.markdown("<h4 class='section-header'>Streak Leaderboard Strategy</h4>", unsafe_allow_html=True)
            streak_scenario = create_streak_scenario(selected_bogger, metrics_df)
            st.markdown(f"""
            <div class='info-box'>
                <p class='dark-text'>{streak_scenario}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<h4 class='section-header'>10K Days Leaderboard Strategy</h4>", unsafe_allow_html=True)
            days_scenario = create_10k_days_scenario(selected_bogger, metrics_df)
            st.markdown(f"""
            <div class='info-box'>
                <p class='dark-text'>{days_scenario}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add "no shame" message appropriately
            bogger_streak_rank = metrics_df[metrics_df['Name'] == selected_bogger].iloc[0]['streak_rank']
            bogger_days_rank = metrics_df[metrics_df['Name'] == selected_bogger].iloc[0]['days_rank']
            
            # Find the first place people
            first_place_streak = metrics_df[metrics_df['streak_rank'] == 1].iloc[0]['Name']
            first_place_days = metrics_df[metrics_df['days_rank'] == 1].iloc[0]['Name']
            
            # Customize message based on their rankings
            message = ""
            
            # Case 1: If they're #1 in both challenges, no message
            if bogger_streak_rank == 1 and bogger_days_rank == 1:
                pass  # No message
                
            # Case 2: If they're #1 in streak but not in 10K days
            elif bogger_streak_rank == 1:
                message = f"There's no shame in losing to {first_place_days} in 10K days."
                
            # Case 3: If they're #1 in 10K days but not in streak
            elif bogger_days_rank == 1:
                message = f"There's no shame in losing to {first_place_streak} in streaks."
                
            # Case 4: If they're not #1 in either (same person leads both)
            elif first_place_streak == first_place_days:
                message = f"There's no shame in losing to {first_place_streak}."
                
            # Case 5: If they're not #1 in either (different leaders)
            else:
                message = f"There's no shame in losing to {first_place_streak} in streaks and {first_place_days} in 10K days."
            
            # Display the message if we have one
            if message:
                st.markdown(f"""
                <div class='info-box'>
                    <p class='dark-text'>{message}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Best week analysis with mobile-friendly layout
            st.markdown("<div class='section-spacing'></div>", unsafe_allow_html=True)
            st.markdown("<h3 class='section-header'>Best Week Analysis</h3>", unsafe_allow_html=True)
            
            # Display streak week first, then 10K days week for mobile
            st.markdown("<h4 class='section-header'>Best Streak Week</h4>", unsafe_allow_html=True)
            best_streak_week = find_best_week_streak(df, date_cols, selected_bogger)
            
            if best_streak_week:
                st.markdown(f"""
                <div class='stat-box'>
                    <p class='dark-text'><strong>Best Week:</strong> {best_streak_week['start_date']} to {best_streak_week['end_date']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Longest Streak", f"{best_streak_week['longest_streak']} days")
                    st.metric("Total Steps", f"{best_streak_week['total_steps']:,}")
                with col2:
                    st.metric("Daily Average", f"{int(best_streak_week['avg_steps']):,}")
            else:
                st.markdown("""
                <div class='info-box'>
                    <p class='dark-text'>Not enough data to determine best streak week.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<h4 class='section-header'>Best 10K Days Week</h4>", unsafe_allow_html=True)
            best_10k_week = find_best_week_10k(df, date_cols, selected_bogger)
            
            if best_10k_week:
                st.markdown(f"""
                <div class='stat-box'>
                    <p class='dark-text'><strong>Best Week:</strong> {best_10k_week['start_date']} to {best_10k_week['end_date']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("10K Days", f"{best_10k_week['days_with_10k']} / 7")
                    st.metric("Total Steps", f"{best_10k_week['total_steps']:,}")
                with col2:
                    st.metric("Daily Average", f"{int(best_10k_week['avg_steps']):,}")
            else:
                st.markdown("""
                <div class='info-box'>
                    <p class='dark-text'>Not enough data to determine best 10K days week.</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<h2 class='section-header'>Weekly MVPs</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>Highlighting top performers for each week of the challenge</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate weekly MVPs
        weekly_mvps = calculate_weekly_mvps(df, date_cols)
        
        if weekly_mvps:
            # Display MVPs in cards - stacked for mobile
            for mvp in weekly_mvps:
                week_range = f"{mvp['start_date']} to {mvp['end_date']}"
                
                # Create expandable section for each week
                with st.expander(f"Week {mvp['week_num']}: {week_range}", expanded=mvp['week_num'] == 1):
                    # Check if this is a future week
                    if mvp.get('future_week', False):
                        st.markdown("""
                        <div class='future-week'>
                            <h4>📅 Week Not Yet Available</h4>
                            <p>Data for this week will be displayed when available.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Mobile-friendly: stack MVPs vertically
                        st.markdown(f"""
                        <div class='mvp-card'>
                            <h4 class='dark-text'>Longest Streak MVP{" (Tie)" if mvp['streak_mvp'].get('is_tie', False) else ""}</h4>
                            <p style='font-size: 1.2rem;' class='dark-text'>{mvp['streak_mvp']['name']}</p>
                            <p class='dark-text'>{mvp['streak_mvp']['streak']} consecutive days</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class='mvp-card'>
                            <h4 class='dark-text'>Most 10K Days MVP{" (Tie)" if mvp['days_mvp'].get('is_tie', False) else ""}</h4>
                            <p style='font-size: 1.2rem;' class='dark-text'>{mvp['days_mvp']['name']}</p>
                            <p class='dark-text'>{mvp['days_mvp']['days']} days with 10K+ steps</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <p class='dark-text'>Not enough data to calculate weekly MVPs.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("<h2 class='section-header'>Community Statistics</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>Tracking our collective progress in the challenge</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate community stats
        community_stats = generate_community_stats(df, date_cols, available_end_date_str)
        
        # Mobile-friendly stats display - Stack vertically
        st.markdown(f"""
        <div class='stat-box'>
            <h3 class='dark-text'>Total Distance</h3>
            <p style='font-size: 1.5rem;' class='dark-text'>{community_stats['total_distance_km']:,.1f} km</p>
        </div>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
        <div class='stat-box'>
            <h3 class='dark-text'>Cosmic Progress</h3>
            <p style='font-size: 1.5rem;' class='dark-text'>{community_stats['milestone_percentage']:.1f}%</p>
            <p class='dark-text'>{community_stats['cosmic_milestone']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add marathons for mobile users
        st.markdown(f"""
        <div class='stat-box'>
            <h3 class='dark-text'>Marathon Equivalent</h3>
            <p style='font-size: 1.5rem;' class='dark-text'>{community_stats['marathons']:.1f}</p>
            <p class='dark-text'>Each marathon is 42.195 km</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Display welcome message when no data is available - using single quotes and more compact HTML
    st.markdown('<div class="stat-box"><h2 class="dark-text">Welcome to the 84-Day Boggers Step Challenge! 🦦</h2><p class="dark-text">This is your central hub for tracking progress in our step challenge.</p></div>', unsafe_allow_html=True)

    # Check if we're pre-challenge or if there's just no data file
    if DEFAULT_CSV_PATH is None:
        # No CSV found - challenge may not have started yet
        
        # Get today's date
        today = date.today()
        
        # Fixed start date: April 10th
        challenge_start = datetime.datetime(2025, 4, 10).date()
        days_to_start = (challenge_start - today).days
        
        if days_to_start > 0:
            # Challenge hasn't started yet
            st.markdown(f'<div class="info-box"><h3 class="dark-text">🗓️ Challenge Countdown</h3><p style="font-size: 1.8rem; text-align: center;" class="dark-text">{days_to_start} days to go!</p><p class="dark-text">The challenge begins on {challenge_start.strftime("%A, %B %d, %Y")}.</p></div>', unsafe_allow_html=True)
        else:
            # Challenge has started but no data uploaded yet
            st.markdown('<div class="info-box"><h3 class="dark-text">🏃‍♀️ Challenge In Progress</h3><p class="dark-text">The challenge has started but Batsi hasn\'t updated the website yet.</p><p class="dark-text">Keep stepping and check back soon!</p></div>', unsafe_allow_html=True)
    
    # Challenge details explanation
    st.markdown('<div class="info-box"><h3 class="dark-text">Challenge Overview:</h3><ul><li class="dark-text"><strong>Duration</strong>: 84 days (12 weeks)</li><li class="dark-text"><strong>Main Goals</strong>:<ul><li class="dark-text">Build a streak of consecutive 10K+ step days</li><li class="dark-text">Accumulate as many 10K+ step days as possible</li></ul></li><li class="dark-text"><strong>Rankings</strong>: Two separate leaderboards - longest streak and most 10K days</li></ul></div>', unsafe_allow_html=True)

# Create a section header
st.markdown("<h3 class='section-header'>❓ Frequently Asked Questions</h3>", unsafe_allow_html=True)

# Create the FAQ section using the most minimal HTML possible
st.markdown("""
<div class="info-box">
<p><b>Q: How do I join the competition?</b><br>
A: It's simple:<br>
1. First, get the StepUp app on your phone<br>
2. Connect it to the health app on your phone<br>
3. Join the Boggers group using this link: https://join.thestepupapp.com/w83T</p>

<p><b>Q: What happens if there's a tie in the streak/most 10k steps leaderboard?</b><br>
A: If two or more boggers have the same streak length or the same number of 10k days, the tiebreaker will be determined by total number of steps. The bogger with the highest total steps wins.</p>

<p><b>Q: What's the prize for winning?</b><br>
A: Bragging rights!</p>

<p><b>Q: When will the leaderboard/website be updated?</b><br>
A: Once a week until the competition is over. Note that I might forget to do so. So just remind me if I forget.</p>

<p><b>Q: When is Arsenal playing Real Madrid in the UCL quarter finals?</b><br>
A: First leg 8th April 9 pm SAST, Second leg 16th April 9pm SAST</p>

<p><b>Q: Batsi don't you have a thesis to write?</b><br>
A: You're goddamn right I do. But I wanted to procrastinate.</p>
</div>
""", unsafe_allow_html=True)