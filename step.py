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
    st.session_state.challenge_start_date = None
if 'challenge_end_date' not in st.session_state:
    st.session_state.challenge_end_date = None
if 'view_start_date' not in st.session_state:
    st.session_state.view_start_date = None
if 'view_end_date' not in st.session_state:
    st.session_state.view_end_date = None
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
MILESTONE_DAYS = [week * 7 for week in MILESTONE_WEEKS]  # Convert weeks to days: 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84

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

# Function to calculate streaks and metrics - UPDATED to use weekly milestones
def calculate_metrics(df, date_cols, start_date_str, end_date_str):
    # Convert date strings to datetime
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    
    # Filter date columns within the selected range
    filtered_date_cols = [col for col in date_cols if start_date <= datetime.datetime.strptime(col, '%Y-%m-%d').date() <= end_date]
    
    # Initialize results dataframe with appropriate data types
    results = pd.DataFrame()
    results['Name'] = df['Name']
    
    # Use the correct data types from the start
    results['days_with_10k'] = 0
    results['current_streak'] = 0
    results['longest_streak'] = 0
    results['total_steps'] = 0
    results['total_distance_km'] = 0.0  # Initialize as float instead of int
    results['highest_milestone'] = 0  # Track highest milestone for each bogger
    
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
        total_distance_km = 0.0
        
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
                    total_distance_km = 0.0
            
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
        
        # Check for new streak milestones - UPDATED to use weekly milestones
        if int(longest_streak) > int(prev_longest_streak):
            # Check for milestone crossings
            for milestone in MILESTONE_DAYS:
                if int(prev_longest_streak) < milestone <= int(longest_streak):
                    # Record new milestone
                    if name not in st.session_state.milestones:
                        st.session_state.milestones[name] = []
                    
                    # Add milestone if not already there
                    if milestone not in st.session_state.milestones[name]:
                        st.session_state.milestones[name].append(milestone)
                        
                        # Set milestone notification - Convert days to weeks for display
                        weeks = milestone // 7
                        st.session_state.show_milestone = True
                        st.session_state.milestone_message = f"🎉 Amazing achievement! {name} has reached a {weeks}-week streak!"
                        st.session_state.milestone_bogger = name
                        st.session_state.milestone_days = milestone
        
        # Set highest milestone for this bogger (for leaderboard display)
        if name in st.session_state.milestones and st.session_state.milestones[name]:
            results.loc[idx, 'highest_milestone'] = max(st.session_state.milestones[name])
        else:
            results.loc[idx, 'highest_milestone'] = 0
        
        # Update previous longest streak
        prev_longest_streaks[name] = int(longest_streak)
    
    # Custom ranking for longest streak with total steps as tiebreaker
    # First sort by longest streak (descending) and then by total steps (descending) for ties
    results_sorted_by_streak = results.sort_values(['longest_streak', 'total_steps'], ascending=[False, False])
    results_sorted_by_streak['streak_rank'] = range(1, len(results_sorted_by_streak) + 1)

    # Custom ranking for most 10K days with total steps as tiebreaker
    # First sort by days with 10k (descending) and then by total steps (descending) for ties
    results_sorted_by_days = results.sort_values(['days_with_10k', 'total_steps'], ascending=[False, False])
    results_sorted_by_days['days_rank'] = range(1, len(results_sorted_by_days) + 1)

    # Update the ranks in the original results DataFrame
    for idx, row in results.iterrows():
        name = row['Name']
        results.loc[idx, 'streak_rank'] = results_sorted_by_streak[results_sorted_by_streak['Name'] == name]['streak_rank'].values[0]
        results.loc[idx, 'days_rank'] = results_sorted_by_days[results_sorted_by_days['Name'] == name]['days_rank'].values[0]
    
    # Check for perfect streak - UPDATED to only consider perfect from challenge start
    results['perfect_streak_potential'] = False
    
    if st.session_state.challenge_start_date:
        today = date.today()
        st.session_state.perfect_boggers = []
        
        for _, row in results.iterrows():
            name = row['Name']
            # Check if bogger has a perfect streak from challenge start until today
            if has_perfect_streak(df, name, st.session_state.challenge_start_date, today):
                results.loc[results['Name'] == name, 'perfect_streak_potential'] = True
                if name not in st.session_state.perfect_boggers:
                    st.session_state.perfect_boggers.append(name)
    
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

# Function to calculate weekly MVPs and leaderboards
def calculate_weekly_stats(df, date_cols):
    # Group date columns by week
    weeks = []
    current_week = []
    
    for i, col in enumerate(date_cols):
        current_week.append(col)
        # If we have 7 days or we're at the end of the date columns
        if len(current_week) == 7 or i == len(date_cols) - 1:
            weeks.append(current_week)
            current_week = []
    
    # Calculate stats for each week
    weekly_stats = []
    
    for i, week_cols in enumerate(weeks):
        week_start = datetime.datetime.strptime(week_cols[0], '%Y-%m-%d').date()
        week_end = (week_start + timedelta(days=len(week_cols) - 1))
        
        # Check if this week is in the future
        if week_end > date.today():
            weekly_stats.append({
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
        week_metrics['highest_milestone'] = 0  # Add milestone tracking
        week_metrics['perfect_streak_potential'] = False  # Add perfect streak tracking
        
        # Calculate metrics for this week
        for idx, row in df.iterrows():
            days_with_10k = 0
            current_streak = 0
            longest_streak = 0
            total_steps = 0
            name = row['Name']
            
            # Check if this bogger has a perfect streak for just this week
            has_perfect_streak_this_week = True
            
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
                    has_perfect_streak_this_week = False  # If any day is < 10K, not perfect
            
            # Store the metrics
            week_metrics.loc[idx, 'days_with_10k'] = days_with_10k
            week_metrics.loc[idx, 'longest_streak'] = longest_streak
            week_metrics.loc[idx, 'total_steps'] = total_steps
            week_metrics.loc[idx, 'perfect_streak_potential'] = has_perfect_streak_this_week
            
            # Apply any global milestones this bogger has achieved
            if name in st.session_state.milestones and st.session_state.milestones[name]:
                week_metrics.loc[idx, 'highest_milestone'] = max(st.session_state.milestones[name])
            else:
                week_metrics.loc[idx, 'highest_milestone'] = 0
        
        # Calculate rankings
        streak_rankings = week_metrics.sort_values(['longest_streak', 'total_steps'], ascending=[False, False])
        streak_rankings['streak_rank'] = range(1, len(streak_rankings) + 1)
        
        days_rankings = week_metrics.sort_values(['days_with_10k', 'total_steps'], ascending=[False, False])
        days_rankings['days_rank'] = range(1, len(days_rankings) + 1)
        
        # Update ranks in week_metrics
        for idx, row in week_metrics.iterrows():
            name = row['Name']
            week_metrics.loc[idx, 'streak_rank'] = streak_rankings[streak_rankings['Name'] == name]['streak_rank'].values[0]
            week_metrics.loc[idx, 'days_rank'] = days_rankings[days_rankings['Name'] == name]['days_rank'].values[0]
        
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
        
        # Add to weekly stats list
        weekly_stats.append({
            'week_num': i + 1,
            'start_date': week_start.strftime('%Y-%m-%d'),
            'end_date': week_end.strftime('%Y-%m-%d'),
            'streak_mvp': streak_mvp,
            'days_mvp': days_mvp,
            'leaderboards': week_metrics,
            'future_week': False
        })
    
    return weekly_stats

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
    total_distance_km = 0.0
    
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

# Helper function to get the milestone badge HTML - UPDATED to use weeks
def get_milestone_badge(milestone_days):
    if milestone_days <= 0:
        return ""
    weeks = milestone_days // 7
    return f'<span class="milestone-badge">🏆 {weeks} week{"s" if weeks != 1 else ""}</span>'

# Helper function to count milestone achievements - UPDATED for weekly milestones
def get_milestone_counts():
    if not st.session_state.milestones:
        return {}
    
    # Count how many people have reached each milestone
    milestone_counts = {days: 0 for days in MILESTONE_DAYS}
    
    for name, milestones in st.session_state.milestones.items():
        for milestone in milestones:
            if milestone in milestone_counts:
                milestone_counts[milestone] += 1
    
    # Convert days to weeks for display
    return {days_to_weeks(days): count for days, count in milestone_counts.items() if count > 0}

# Function to display milestone hall of fame content
def display_milestone_hall_of_fame():
    if st.session_state.milestones:
        # Get a count of all milestone achievements
        milestone_counts = get_milestone_counts()
        
        # Display milestone summary with weeks instead of days
        milestone_summary = ", ".join([f"{count} boggers reached {milestone}" 
                                     for milestone, count in milestone_counts.items()])
        
        # Calculate who can still get the perfect 84-day milestone
        today = date.today()
        if st.session_state.challenge_start_date and st.session_state.challenge_end_date:
            days_elapsed = (today - st.session_state.challenge_start_date).days + 1
            days_total = (st.session_state.challenge_end_date - st.session_state.challenge_start_date).days + 1
            
            perfect_streak_possible = len(st.session_state.perfect_boggers)
            
            if days_elapsed < days_total:
                st.markdown(f"""
                <div class='info-box'>
                    <p class='dark-text'><strong>Milestone Summary:</strong> {milestone_summary}</p>
                    <p class='dark-text'><strong>Perfect Streak Potential:</strong> {perfect_streak_possible} boggers still have a chance to achieve the perfect 12-week streak!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Challenge is over
                st.markdown(f"""
                <div class='info-box'>
                    <p class='dark-text'><strong>Milestone Summary:</strong> {milestone_summary}</p>
                    <p class='dark-text'><strong>Challenge Complete!</strong> {perfect_streak_possible} boggers achieved the perfect 12-week streak!</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create a grid layout for milestones
        milestone_cols = st.columns(min(3, len(st.session_state.milestones)))
        
        for i, (name, milestones) in enumerate(st.session_state.milestones.items()):
            col_index = i % len(milestone_cols)
            with milestone_cols[col_index]:
                # Convert days to weeks for display
                weeks_display = [f"{days//7} week{'s' if days//7 != 1 else ''}" for days in sorted(milestones)]
                
                st.markdown(f"""
                <div class='stat-box'>
                    <p class='dark-text' style='font-size: 1.2rem;'><strong>{name}</strong></p>
                    <p class='dark-text'>{', '.join(weeks_display)}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>No streak milestones achieved yet! Keep stepping! 🦦</p>
            <p class='dark-text'>Milestones are awarded at 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 weeks.</p>
        </div>
        """, unsafe_allow_html=True)

# Function to display filter controls
def display_filter_controls():
    st.markdown("<h3 class='section-header'>View Filters</h3>", unsafe_allow_html=True)
    
    # Create a filter box
    with st.container():
        st.markdown('<div class="filter-box">', unsafe_allow_html=True)
        
        # Challenge info
        if st.session_state.challenge_start_date:
            # Calculate days left in the challenge
            today = date.today()
            days_left = (st.session_state.challenge_end_date - today).days
            days_left = max(0, days_left)  # Ensure it doesn't go negative
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<p class='dark-text'><strong>Start:</strong> {st.session_state.challenge_start_date.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='dark-text'><strong>Days Left:</strong> {days_left}</p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<p class='dark-text'><strong>End:</strong> {st.session_state.challenge_end_date.strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='dark-text'><strong>Total Days:</strong> 84 (12 weeks)</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Calculate weekly stats for MVPs and weekly leaderboards
    weekly_stats = calculate_weekly_stats(df, date_cols)

    # Create tabs
    tab_names = ["📊 Leaderboards", "🏆 Milestones", "🔍 Details", "🗓️ Weekly Leaderboards", "📅 MVPs", "📈 Stats"]
    tabs = st.tabs(tab_names)
    
    # Track currently selected tab
    with tabs[0]:  # Leaderboards tab
        # Show Ultimate Challenge Goal Banner only on the leaderboards tab
        st.markdown("""
        <div class='milestone-notification'>
            <h3>🏆 Ultimate Challenge Goal: 12-Week Streak! 🏆</h3>
            <p class='dark-text'>Can you maintain a perfect streak for the entire challenge?</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-header'>Step Challenge Leaderboards</h2>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p class='dark-text'><strong>Viewing data from:</strong> {available_start_date_str} to {available_end_date_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mobile-optimized leaderboards - Stack them vertically instead of side by side
        st.markdown("<h3 class='section-header'>Longest Streak Leaderboard</h3>", unsafe_allow_html=True)
        
        # Create the streak leaderboard
        streak_leaderboard = metrics_df[['streak_rank', 'Name', 'longest_streak', 'perfect_streak_potential', 'highest_milestone']].copy()
        streak_leaderboard = streak_leaderboard.sort_values('streak_rank')
        
        # Add crown emoji next to names with perfect streaks and milestone badges
        def add_crown_and_milestone(row):
            name = row['Name']
            has_potential = row['perfect_streak_potential']
            milestone = row['highest_milestone']
            
            display_name = name
            if has_potential and name in st.session_state.perfect_boggers:
                display_name = f"👑 {display_name}"
                
            milestone_badge = get_milestone_badge(milestone)
            if milestone_badge:
                # Use HTML to display the milestone badge
                display_name = f"{display_name} {milestone_badge}"
                
            return display_name
        
        # Apply crown highlighting and milestone badges
        streak_leaderboard['Name'] = streak_leaderboard.apply(add_crown_and_milestone, axis=1)
        
        # Format the leaderboard
        streak_leaderboard = streak_leaderboard.rename(columns={
            'streak_rank': 'Rank',
            'longest_streak': 'Longest Streak'
        })
        
        # Drop unnecessary columns
        streak_leaderboard = streak_leaderboard.drop(['perfect_streak_potential', 'highest_milestone'], axis=1)
        
        # Display the streak leaderboard
        st.markdown(streak_leaderboard.to_html(escape=False, index=False, classes='custom-table'), unsafe_allow_html=True)
        
        st.markdown("<h3 class='section-header'>Most 10K Days Leaderboard</h3>", unsafe_allow_html=True)
        
        # Create the 10K days leaderboard
        days_leaderboard = metrics_df[['days_rank', 'Name', 'days_with_10k', 'perfect_streak_potential', 'highest_milestone']].copy()
        days_leaderboard = days_leaderboard.sort_values('days_rank')
        
        # Apply crown highlighting and milestone badges
        days_leaderboard['Name'] = days_leaderboard.apply(add_crown_and_milestone, axis=1)
        
        # Format the leaderboard
        days_leaderboard = days_leaderboard.rename(columns={
            'days_rank': 'Rank',
            'days_with_10k': '10K Days'
        })
        
        # Drop unnecessary columns
        days_leaderboard = days_leaderboard.drop(['perfect_streak_potential', 'highest_milestone'], axis=1)
        
        # Display the 10K days leaderboard
        st.markdown(days_leaderboard.to_html(escape=False, index=False, classes='custom-table'), unsafe_allow_html=True)
        
        # Add note about crown icons and milestone badges
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>👑 Crown indicates boggers who have maintained a perfect streak from the start of the challenge</p>
            <p class='dark-text'>🏆 Milestone badges show the highest streak milestone achieved (in weeks)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:  # Milestones tab
        # Milestone Hall of Fame has its own tab now
        st.markdown("<h2 class='section-header'>🏆 Milestone Hall of Fame</h2>", unsafe_allow_html=True)
        display_milestone_hall_of_fame()
    
    with tabs[2]:  # Details tab
        st.markdown("""
        <div class='milestone-notification' style="background-color: #5c3c10; border-left-color: #ffc107;">
            <h3>⚠️ Known Issue</h3>
            <p class='dark-text'>When selecting a bogger, the app may return to the Leaderboards tab. 
            If this happens, simply click back to the Details tab to see your selected bogger's information. 
            This will be fixed in a future update.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 class='section-header'>Bogger Details</h2>", unsafe_allow_html=True)
        
        # Improved bogger selection dropdown
        current_bogger = st.session_state.selected_bogger
        if current_bogger is None and len(st.session_state.boggers_names) > 0:
            current_bogger = st.session_state.boggers_names[0]
        
        selected_bogger = st.selectbox(
            "Select Bogger", 
            options=sorted(st.session_state.boggers_names),
            index=sorted(st.session_state.boggers_names).index(current_bogger) if current_bogger in st.session_state.boggers_names else 0,
            key="bogger_selector"
        )
        
        # Store selection in session state
        st.session_state.selected_bogger = selected_bogger
        
        if selected_bogger:
            # Get bogger data
            bogger_data = df[df['Name'] == selected_bogger].iloc[0]
            bogger_metrics = metrics_df[metrics_df['Name'] == selected_bogger].iloc[0]
            
            # Show milestone if any - UPDATED to show weeks instead of days
            if bogger_metrics['highest_milestone'] > 0:
                milestone_weeks = bogger_metrics['highest_milestone'] // 7
                st.markdown(f"""
                <div class='milestone-notification'>
                    <h3>🏆 Milestone Achievement: {milestone_weeks}-Week Streak!</h3>
                </div>
                """, unsafe_allow_html=True)
            
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
    
    with tabs[3]:  # Weekly Leaderboards Tab
        # NEW: Weekly Leaderboards Tab
        st.markdown("<h2 class='section-header'>Weekly Leaderboards</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>See how everyone ranked each week of the challenge</p>
        </div>
        """, unsafe_allow_html=True)
        
        if weekly_stats:
            # Weekly selector
            week_options = [f"Week {stat['week_num']}: {stat['start_date']} to {stat['end_date']}" 
                          for stat in weekly_stats if not stat.get('future_week', False)]
            
            # Default to most recent week
            default_index = len(week_options) - 1
            selected_week_option = st.selectbox("Select Week", options=week_options, index=default_index, key="weekly_selector")
            
            # Get the selected week number
            selected_week_num = int(selected_week_option.split(":")[0].replace("Week ", ""))
            
            # Find the corresponding week stats
            selected_week_stats = next((stat for stat in weekly_stats if stat['week_num'] == selected_week_num), None)
            
            if selected_week_stats and not selected_week_stats.get('future_week', False):
                # Get the leaderboard data
                week_metrics = selected_week_stats['leaderboards']
                
                # Display streak leaderboard for this week
                st.markdown("<h3 class='section-header'>Weekly Streak Leaderboard</h3>", unsafe_allow_html=True)
                
                # Create the streak leaderboard
                streak_board = week_metrics[['streak_rank', 'Name', 'longest_streak', 'perfect_streak_potential', 'highest_milestone']].copy()
                streak_board = streak_board.sort_values('streak_rank')
                
                # Add crown emoji next to names with perfect streak in this week only
                def add_weekly_crown_and_milestone(row):
                    name = row['Name']
                    has_perfect_week = row['perfect_streak_potential']  # For this week only
                    milestone = row['highest_milestone']
                    
                    display_name = name
                    if has_perfect_week:
                        display_name = f"👑 {display_name}"
                        
                    milestone_badge = get_milestone_badge(milestone)
                    if milestone_badge:
                        # Use HTML to display the milestone badge
                        display_name = f"{display_name} {milestone_badge}"
                        
                    return display_name
                
                # Apply crown and milestone badges
                streak_board['Name'] = streak_board.apply(add_weekly_crown_and_milestone, axis=1)
                
                # Format the leaderboard
                streak_board = streak_board.rename(columns={
                    'streak_rank': 'Rank',
                    'longest_streak': 'Longest Streak'
                })
                
                # Drop unnecessary columns
                streak_board = streak_board.drop(['perfect_streak_potential', 'highest_milestone'], axis=1)
                
                # CHANGE HERE: Use markdown with to_html instead of dataframe
                st.markdown(streak_board.to_html(escape=False, index=False, classes='custom-table'), unsafe_allow_html=True)
                
                # Display 10K days leaderboard for this week
                st.markdown("<h3 class='section-header'>Weekly 10K Days Leaderboard</h3>", unsafe_allow_html=True)
                
                # Create the 10K days leaderboard
                days_board = week_metrics[['days_rank', 'Name', 'days_with_10k', 'perfect_streak_potential', 'highest_milestone']].copy()
                days_board = days_board.sort_values('days_rank')
                
                # Apply crown and milestone badges
                days_board['Name'] = days_board.apply(add_weekly_crown_and_milestone, axis=1)
                
                # Format the leaderboard
                days_board = days_board.rename(columns={
                    'days_rank': 'Rank',
                    'days_with_10k': '10K Days'
                })
                
                # Drop unnecessary columns
                days_board = days_board.drop(['perfect_streak_potential', 'highest_milestone'], axis=1)
                
                # CHANGE HERE: Use markdown with to_html instead of dataframe
                st.markdown(days_board.to_html(escape=False, index=False, classes='custom-table'), unsafe_allow_html=True)
                
                # Add note about crown icons and milestone badges
                st.markdown("""
                <div class='info-box'>
                    <p class='dark-text'>👑 Crown indicates boggers who hit 10K steps every day in this week</p>
                    <p class='dark-text'>🏆 Milestone badges show the highest streak milestone achieved (in weeks)</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='info-box'>
                    <p class='dark-text'>Data for this week is not available yet.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <p class='dark-text'>Not enough data to calculate weekly leaderboards.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[4]:  # MVPs tab
        st.markdown("<h2 class='section-header'>Weekly MVPs</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <p class='dark-text'>Highlighting top performers for each week of the challenge</p>
        </div>
        """, unsafe_allow_html=True)
        
        if weekly_stats:
            # Display MVPs in cards - stacked for mobile
            for week_stat in weekly_stats:
                week_range = f"{week_stat['start_date']} to {week_stat['end_date']}"
                
                # Create expandable section for each week
                with st.expander(f"Week {week_stat['week_num']}: {week_range}", expanded=week_stat['week_num'] == 1):
                    # Check if this is a future week
                    if week_stat.get('future_week', False):
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
                            <h4 class='dark-text'>Longest Streak MVP{" (Tie)" if week_stat['streak_mvp'].get('is_tie', False) else ""}</h4>
                            <p style='font-size: 1.2rem;' class='dark-text'>{week_stat['streak_mvp']['name']}</p>
                            <p class='dark-text'>{week_stat['streak_mvp']['streak']} consecutive days</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class='mvp-card'>
                            <h4 class='dark-text'>Most 10K Days MVP{" (Tie)" if week_stat['days_mvp'].get('is_tie', False) else ""}</h4>
                            <p style='font-size: 1.2rem;' class='dark-text'>{week_stat['days_mvp']['name']}</p>
                            <p class='dark-text'>{week_stat['days_mvp']['days']} days with 10K+ steps</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
                <p class='dark-text'>Not enough data to calculate weekly MVPs.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tabs[5]:  # Stats tab
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
3. Join the Boggers group using this link: 
<a href="https://join.thestepupapp.com/w83T" target="_blank">https://join.thestepupapp.com/w83T</a>
</p>


<p><b>Q: What happens if there's a tie in the streak/most 10k steps leaderboard?</b><br>
A: If two or more boggers have the same streak length or the same number of 10k days, the tiebreaker will be determined by total number of steps. The bogger with the highest total steps wins.</p>

<p><b>Q: What's the prize for winning?</b><br>
A: Bragging rights!</p>

<p><b>Q: When will the leaderboard/website be updated?</b><br>
A: Once a week until the competition is over. Note that I might forget to do so. So just remind me if I forget.</p>

<p><b>Q: When is Paris Saint-Germain playing Arsenal in the UCL semi-finals?</b><br>
A: Arsenal lost the first leg 1-0, Second leg 7th May 9pm SAST</p>

<p><b>Q: Batsi don't you have a thesis to write?</b><br>
A: You're goddamn right I do. But I wanted to procrastinate.</p>
</div>
""", unsafe_allow_html=True)