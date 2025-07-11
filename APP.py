import pandas as pd
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import uuid


# Google Sheets auth setup
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gspread"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(credentials)
    return client.open("usage_log").sheet1  # Make sure "usage_log" is the correct name


if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]



def log_to_sheet(action, league="", home="", away=""):

    try:
        sheet = get_gsheet()
        row = [
            str(datetime.datetime.now()),                      # Time
            st.session_state.get("session_id", ""),            # Session ID
            league,
            home,
            away,
            action
        ]
        sheet.append_row(row)

    except Exception as e:
        st.warning(f"⚠️ Logging failed: {e}")



st.set_page_config(
    page_title="The Betting Engineer",
    page_icon="⚙️",
    layout="wide"
)

st.image("header1.png", use_container_width=True)





# Import from league modules
from leagues.NORWAY import (
    load_norway_model, get_last_n_h2h, plot_last_matches_goals_dual,
    get_team_goal_averages,
)
from leagues.SWEDEN import (
    load_sweden_model, get_last_n_h2h as h2h_sweden,
    plot_last_matches_goals_dual as plot_sweden,
    get_team_goal_averages as avg_sweden,

)
from leagues.FINLAND import (
    load_finland_model, get_last_n_h2h as h2h_finland,
    plot_last_matches_goals_dual as plot_finland,
    get_team_goal_averages as avg_finland,

)
from leagues.IRELAND import (
    load_ireland_model, get_last_n_h2h as h2h_ireland,
    plot_last_matches_goals_dual as plot_ireland,
    get_team_goal_averages as avg_ireland,
)

# Sidebar league selector
# League selector and interaction tracker
default_league = "Norway"
league = st.selectbox("Select League", ["Norway", "Sweden", "Finland", "Ireland"])
st.session_state["league"] = league

# Only log once per session or every 30 mins, and only after user interaction
now = datetime.datetime.now()
if "last_log_time" not in st.session_state:
    st.session_state["last_log_time"] = now
    st.session_state["app_open_logged"] = False
    st.session_state["user_interacted"] = False

# Detect if user changed league from default
if league != default_league and not st.session_state["user_interacted"]:
    st.session_state["user_interacted"] = True

# Log only after user interaction (e.g., changed league), and once every 30 minutes
time_diff = (now - st.session_state["last_log_time"]).total_seconds()
should_log = st.session_state["user_interacted"] and (
    not st.session_state["app_open_logged"] or time_diff > 1800
)

if should_log:
    try:
        log_to_sheet("App Opened", league)
        st.session_state["last_log_time"] = now
        st.session_state["app_open_logged"] = True
    except Exception as e:
        st.warning(f"⚠️ Logging failed: {e}")



# Load models and data
if league == "Norway":
    goal_model, df, match_goal_model = load_norway_model()
    get_h2h = get_last_n_h2h
    plot_goals = plot_last_matches_goals_dual
    get_avg = get_team_goal_averages
elif league == "Sweden":
    goal_model, df, match_goal_model = load_sweden_model()
    get_h2h = h2h_sweden
    plot_goals = plot_sweden
    get_avg = avg_sweden

elif league == "Ireland":
    goal_model, df, match_goal_model = load_ireland_model()
    get_h2h = h2h_ireland
    plot_goals = plot_ireland
    get_avg = avg_ireland
else:
    goal_model, df, match_goal_model = load_finland_model()
    get_h2h = h2h_finland
    plot_goals = plot_finland
    get_avg = avg_finland



# Team selection
teams = sorted(set(df["Home"].unique()) | set(df["Away"].unique()))
home = st.selectbox("Home Team", teams, key=f"{league}_home")
away = st.selectbox("Away Team", teams, key=f"{league}_away")
# ❗ Warn if same team selected
if home == away:
    st.warning("⚠️ Please select two different teams.")
    st.stop()
# Button row
# League-specific disclaimer about promoted teams
if league == "Norway":
    st.warning( "⚠️ Info about promoted teams (Bryne, Valerenga) might be inaccurate")
elif league == "Sweden":
    st.warning( "⚠️ Info about promoted teams (Degerfors, Oster)  might be inaccurate")
elif league == "Finland":
    st.warning( "⚠️ Info about promoted teams (KTP, Jaro) might be inaccurate")
elif league == "Ireland":
    st.warning("⚠️ Info about promoted teams (Cork City) might be inaccurate")




col1, col2, col3, col4, col5 = st.columns(5)
show_prediction = show_h2h = show_last15 = show_league_avg = show_match_goals = False

with col1:
    if st.button("Predict Team Goals", key=f"{league}_predict"):
        show_prediction = True
        log_to_sheet("Team Goal Prediction", league, home, away)

with col2:
    if st.button("Predict Match Goals", key=f"{league}_matchgoals"):
        show_match_goals = True
        log_to_sheet("Match Goal Prediction", league, home, away)

with col3:
    if st.button("Last H2H", key=f"{league}_h2h"):
        show_h2h = True
        log_to_sheet("Last H2H", league, home, away)

with col4:
    if st.button("Last 15 Games", key=f"{league}_last15"):
        show_last15 = True
        log_to_sheet("Last 15 Games", league, home, away)

with col5:
    if st.button("League Averages", key=f"{league}_avg"):
        show_league_avg = True
        log_to_sheet("League Averages", league, home, away)


# Display output sections
if show_prediction:
    st.markdown("### Goal Probability Prediction")
    result = goal_model(home, away)
    st.dataframe(result)

if show_h2h:
    st.markdown("### Last Head-to-Head Results")
    h2h_df = get_h2h(home, away, n=3, df=df)

    # Convert 'Date' to datetime and sort
    h2h_df["Date"] = pd.to_datetime(h2h_df["Date"], dayfirst=True)
    h2h_df = h2h_df.sort_values("Date", ascending=False).reset_index(drop=True)
    h2h_df["Date"] = h2h_df["Date"].dt.strftime("%d/%m/%Y")  # Format nicely

    # Show table without index
    st.dataframe(h2h_df, use_container_width=True, hide_index=True)


if show_last15:
    st.markdown("### Last 15 Matches – Goals Scored & Conceded")
    fig = plot_goals(df, home, away, num_matches=15)
    st.pyplot(fig)

if show_league_avg:
    st.markdown("### Average Goals Scored & Conceded per Team")
    league_avg_df = get_avg(df)
    st.dataframe(league_avg_df)

if show_match_goals:
    st.markdown("### Match-Level Goal Probabilities")
    match_df = match_goal_model(home, away)
    st.dataframe(match_df)
