import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# streamlit run APP.py


def load_norway_model():
    # Load data
    pl_df = pd.read_csv("data/NOR.csv")
    elo_df = pd.read_csv("data/rating_NOR.csv")

    # Team replacements
    pl_df["Home"] = pl_df["Home"].replace("Odd", "Bryne")
    pl_df["Away"] = pl_df["Away"].replace("Odd", "Bryne")
    pl_df["Home"] = pl_df["Home"].replace("Lillestrom", "Valerenga")
    pl_df["Away"] = pl_df["Away"].replace("Lillestrom", "Valerenga")

    # Team categories
    def categorize_team(rating):
        if rating <= 4:
            return "Top"
        elif rating <= 8:
            return "UpperMid"
        elif rating <= 12:
            return "LowerMid"
        else:
            return "Bottom"

    elo_df['Category'] = elo_df['Rating'].apply(categorize_team)

    # Merge categories
    pl_df = pl_df.merge(elo_df[['Team', 'Category']], left_on='Home', right_on='Team', how='left')
    pl_df = pl_df.rename(columns={'Category': 'HomeCategory'}).drop('Team', axis=1)
    pl_df = pl_df.merge(elo_df[['Team', 'Category']], left_on='Away', right_on='Team', how='left')
    pl_df = pl_df.rename(columns={'Category': 'AwayCategory'}).drop('Team', axis=1)

    pl_df = pl_df.tail(240)

    # Labels
    for g in [1, 2, 3]:
        pl_df[f"Home_{g}+"] = (pl_df["HG"] >= g).astype(int)
        pl_df[f"Away_{g}+"] = (pl_df["AG"] >= g).astype(int)

    # Feature groups
    team_vs_opp_cat = pl_df.groupby(["Home", "AwayCategory"])["HG"].mean().reset_index()
    team_vs_opp_cat.columns = ["Team", "OpponentCategory", "TeamVsOppCat"]

    opp_cat_concede_vs_team = pl_df.groupby(["AwayCategory", "Home"])["HG"].mean().reset_index()
    opp_cat_concede_vs_team.columns = ["OpponentCategory", "Team", "OppCatConcedeVsTeam"]

    cat_vs_opp = pl_df.groupby(["HomeCategory", "Away"])["HG"].mean().reset_index()
    cat_vs_opp.columns = ["Category", "Opponent", "CatVsOpp"]

    opp_concede_vs_cat = pl_df.groupby(["Away", "HomeCategory"])["HG"].mean().reset_index()
    opp_concede_vs_cat.columns = ["Team", "OpponentCategory", "OppConcedeVsCat"]

    # Generate training rows
    def get_safe(df, key1, val1, key2, val2, col):
        try:
            return df[(df[key1] == val1) & (df[key2] == val2)][col].values[0]
        except:
            return 1.0

    home_rows_ext = []
    away_rows_ext = []

    for _, row in pl_df.iterrows():
        home, away = row["Home"], row["Away"]
        hc, ac = row["HomeCategory"], row["AwayCategory"]

        home_rows_ext.append({
            "TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", home, "OpponentCategory", ac, "TeamVsOppCat"),
            "OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", home, "OpponentCategory", ac, "OppCatConcedeVsTeam"),
            "CatVsOpp": get_safe(cat_vs_opp, "Category", hc, "Opponent", away, "CatVsOpp"),
            "OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", away, "OpponentCategory", hc, "OppConcedeVsCat"),
            "1+": row["Home_1+"],
            "2+": row["Home_2+"],
            "3+": row["Home_3+"],
        })

        away_rows_ext.append({
            "TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", away, "OpponentCategory", hc, "TeamVsOppCat"),
            "OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", away, "OpponentCategory", hc, "OppCatConcedeVsTeam"),
            "CatVsOpp": get_safe(cat_vs_opp, "Category", ac, "Opponent", home, "CatVsOpp"),
            "OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", home, "OpponentCategory", ac, "OppConcedeVsCat"),
            "1+": row["Away_1+"],
            "2+": row["Away_2+"],
            "3+": row["Away_3+"],
        })

    home_df_ext = pd.DataFrame(home_rows_ext)
    away_df_ext = pd.DataFrame(away_rows_ext)

    # Train models
    features_ext = ["TeamVsOppCat", "OppCatConcedeVsTeam", "CatVsOpp", "OppConcedeVsCat"]
    home_models_ext = {}
    away_models_ext = {}

    for label in ["1+", "2+", "3+"]:
        clf_home = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
        home_model = CalibratedClassifierCV(clf_home, method='isotonic', cv=3)
        home_model.fit(home_df_ext[features_ext], home_df_ext[label])
        home_models_ext[label] = home_model

        clf_away = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
        away_model = CalibratedClassifierCV(clf_away, method='isotonic', cv=3)
        away_model.fit(away_df_ext[features_ext], away_df_ext[label])
        away_models_ext[label] = away_model

    # Create prediction function
    def goal_probabilities_df(home_team, away_team):
        home_cat = elo_df[elo_df["Team"] == home_team]["Category"].values[0]
        away_cat = elo_df[elo_df["Team"] == away_team]["Category"].values[0]

        def safe_get(df, key1, val1, key2, val2, col):
            try:
                return df[(df[key1] == val1) & (df[key2] == val2)][col].values[0]
            except:
                return 1.0

        X_home = pd.DataFrame([{
            "TeamVsOppCat": safe_get(team_vs_opp_cat, "Team", home_team, "OpponentCategory", away_cat, "TeamVsOppCat"),
            "OppCatConcedeVsTeam": safe_get(opp_cat_concede_vs_team, "Team", home_team, "OpponentCategory", away_cat, "OppCatConcedeVsTeam"),
            "CatVsOpp": safe_get(cat_vs_opp, "Category", home_cat, "Opponent", away_team, "CatVsOpp"),
            "OppConcedeVsCat": safe_get(opp_concede_vs_cat, "Team", away_team, "OpponentCategory", home_cat, "OppConcedeVsCat")
        }])

        X_away = pd.DataFrame([{
            "TeamVsOppCat": safe_get(team_vs_opp_cat, "Team", away_team, "OpponentCategory", home_cat, "TeamVsOppCat"),
            "OppCatConcedeVsTeam": safe_get(opp_cat_concede_vs_team, "Team", away_team, "OpponentCategory", home_cat, "OppCatConcedeVsTeam"),
            "CatVsOpp": safe_get(cat_vs_opp, "Category", away_cat, "Opponent", home_team, "CatVsOpp"),
            "OppConcedeVsCat": safe_get(opp_concede_vs_cat, "Team", home_team, "OpponentCategory", away_cat, "OppConcedeVsCat")
        }])

        def format_percent(prob):
            return f"{round(prob * 100, 1)}%"

        return pd.DataFrame([
            {"Team": home_team, "Side": "Home",
             "1+": format_percent(home_models_ext["1+"].predict_proba(X_home)[0][1]),
             "2+": format_percent(home_models_ext["2+"].predict_proba(X_home)[0][1]),
             "3+": format_percent(home_models_ext["3+"].predict_proba(X_home)[0][1])},
            {"Team": away_team, "Side": "Away",
             "1+": format_percent(away_models_ext["1+"].predict_proba(X_away)[0][1]),
             "2+": format_percent(away_models_ext["2+"].predict_proba(X_away)[0][1]),
             "3+": format_percent(away_models_ext["3+"].predict_proba(X_away)[0][1])},
        ])

    match_model = load_match_goal_model(pl_df, elo_df)
    return goal_probabilities_df, pl_df, match_model


def get_last_n_h2h(home_team, away_team, n, df):
    mask = (
        ((df["Home"] == home_team) & (df["Away"] == away_team)) |
        ((df["Home"] == away_team) & (df["Away"] == home_team))
    )
    h2h_matches = df[mask].sort_values(by="Date", ascending=False)
    return h2h_matches.head(n)[["Date", "Home", "Away", "HG", "AG", "Res"]]


import matplotlib.pyplot as plt

def plot_last_matches_goals_dual(df, home_team, away_team, num_matches=10):
    import matplotlib.pyplot as plt

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    def prepare_team_data(team):
        matches = df[(df["Home"] == team) | (df["Away"] == team)].copy()
        matches = matches.sort_values("Date", ascending=False).head(num_matches)

        data = []
        for _, row in matches.iterrows():
            is_home = row["Home"] == team
            opponent = row["Away"] if is_home else row["Home"]
            label = f"{opponent[:3].upper()}\n{row['Date'].strftime('%d/%m')}"
            goals_for = row["HG"] if is_home else row["AG"]
            goals_against = row["AG"] if is_home else row["HG"]
            data.append((label, goals_for, goals_against))

        return pd.DataFrame(data, columns=["Opponent", "GoalsFor", "GoalsAgainst"])

    df_home = prepare_team_data(home_team)
    df_away = prepare_team_data(away_team)

    # 📱 Smaller figsize for better mobile rendering
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))  # From (11, 8) to (7, 6)

    def plot_bar(ax, df_team, team):
        x = range(len(df_team))
        ax.bar([i - 0.2 for i in x], df_team["GoalsFor"], width=0.4, label="Scored", color="orange")
        ax.bar([i + 0.2 for i in x], df_team["GoalsAgainst"], width=0.4, label="Conceded", color="gray")

        for i, v in enumerate(df_team["GoalsFor"]):
            ax.text(i - 0.2, v + 0.1, f"{v:.0f}", ha="center", fontsize=7)
        for i, v in enumerate(df_team["GoalsAgainst"]):
            ax.text(i + 0.2, v + 0.1, f"{v:.0f}", ha="center", fontsize=7)

        ax.axhline(df_team["GoalsFor"].mean(), color="green", linestyle="--", linewidth=1,
                   label=f"Avg Scored ({df_team['GoalsFor'].mean():.2f})")
        ax.axhline(df_team["GoalsAgainst"].mean(), color="black", linestyle="--", linewidth=1,
                   label=f"Avg Conceded ({df_team['GoalsAgainst'].mean():.2f})")

        ax.set_xticks(x)
        ax.set_xticklabels(df_team["Opponent"], fontsize=8)
        ax.set_title(f"{team} – Last {num_matches} Matches", fontsize=11)
        ax.legend(fontsize=8)

    plot_bar(ax1, df_home, home_team)
    plot_bar(ax2, df_away, away_team)

    plt.tight_layout()
    return fig


def get_team_goal_averages(df):
    home_stats = df.groupby("Home").agg(
        Matches_Home=("HG", "count"),
        Goals_Scored_Home=("HG", "sum"),
        Goals_Conceded_Home=("AG", "sum")
    )

    away_stats = df.groupby("Away").agg(
        Matches_Away=("AG", "count"),
        Goals_Scored_Away=("AG", "sum"),
        Goals_Conceded_Away=("HG", "sum")
    )

    team_stats = home_stats.join(away_stats, how="outer").fillna(0)

    team_stats["Total_Matches"] = team_stats["Matches_Home"] + team_stats["Matches_Away"]
    team_stats["Goals_Scored"] = team_stats["Goals_Scored_Home"] + team_stats["Goals_Scored_Away"]
    team_stats["Goals_Conceded"] = team_stats["Goals_Conceded_Home"] + team_stats["Goals_Conceded_Away"]

    team_stats["Avg_Scored"] = (team_stats["Goals_Scored"] / team_stats["Total_Matches"]).round(2)
    team_stats["Avg_Conceded"] = (team_stats["Goals_Conceded"] / team_stats["Total_Matches"]).round(2)

    return team_stats[["Avg_Scored", "Avg_Conceded"]].sort_values(by="Avg_Scored", ascending=False)

def load_match_goal_model(df, elo_df):
    # Add goal thresholds
    df["TotalGoals"] = df["HG"] + df["AG"]
    for n in range(1, 7):
        df[f"Goals_{n}+"] = (df["TotalGoals"] >= n).astype(int)

    # Group features (home-based)
    team_vs_opp_cat = df.groupby(["Home", "AwayCategory"])["HG"].mean().reset_index()
    team_vs_opp_cat.columns = ["Team", "OpponentCategory", "TeamVsOppCat"]

    opp_cat_concede_vs_team = df.groupby(["AwayCategory", "Home"])["HG"].mean().reset_index()
    opp_cat_concede_vs_team.columns = ["OpponentCategory", "Team", "OppCatConcedeVsTeam"]

    cat_vs_opp = df.groupby(["HomeCategory", "Away"])["HG"].mean().reset_index()
    cat_vs_opp.columns = ["Category", "Opponent", "CatVsOpp"]

    opp_concede_vs_cat = df.groupby(["Away", "HomeCategory"])["HG"].mean().reset_index()
    opp_concede_vs_cat.columns = ["Team", "OpponentCategory", "OppConcedeVsCat"]

    def get_safe(df, key1, val1, key2, val2, col):
        try:
            return df[(df[key1] == val1) & (df[key2] == val2)][col].values[0]
        except:
            return 1.0

    # Training rows
    rows = []
    for _, row in df.iterrows():
        home, away = row["Home"], row["Away"]
        hc, ac = row["HomeCategory"], row["AwayCategory"]

        features = {
            "Home_TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", home, "OpponentCategory", ac, "TeamVsOppCat"),
            "Home_OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", home, "OpponentCategory", ac, "OppCatConcedeVsTeam"),
            "Home_CatVsOpp": get_safe(cat_vs_opp, "Category", hc, "Opponent", away, "CatVsOpp"),
            "Home_OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", away, "OpponentCategory", hc, "OppConcedeVsCat"),
            "Away_TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", away, "OpponentCategory", hc, "TeamVsOppCat"),
            "Away_OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", away, "OpponentCategory", hc, "OppCatConcedeVsTeam"),
            "Away_CatVsOpp": get_safe(cat_vs_opp, "Category", ac, "Opponent", home, "CatVsOpp"),
            "Away_OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", home, "OpponentCategory", ac, "OppConcedeVsCat"),
        }
        for n in range(1, 7):
            features[f"Goals_{n}+"] = row[f"Goals_{n}+"]
        rows.append(features)

    match_df = pd.DataFrame(rows)
    features_comb = [col for col in match_df.columns if col.startswith("Home_") or col.startswith("Away_")]

    goal_models = {}
    for n in range(1, 7):
        label = f"Goals_{n}+"
        model = CalibratedClassifierCV(
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=0),
            method='isotonic', cv=3
        )
        model.fit(match_df[features_comb], match_df[label])
        goal_models[label] = model

    # Return prediction function
    def match_goal_probabilities(home_team, away_team):
        home_cat = elo_df[elo_df["Team"] == home_team]["Category"].values[0]
        away_cat = elo_df[elo_df["Team"] == away_team]["Category"].values[0]

        X = pd.DataFrame([{
            "Home_TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", home_team, "OpponentCategory", away_cat, "TeamVsOppCat"),
            "Home_OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", home_team, "OpponentCategory", away_cat, "OppCatConcedeVsTeam"),
            "Home_CatVsOpp": get_safe(cat_vs_opp, "Category", home_cat, "Opponent", away_team, "CatVsOpp"),
            "Home_OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", away_team, "OpponentCategory", home_cat, "OppConcedeVsCat"),
            "Away_TeamVsOppCat": get_safe(team_vs_opp_cat, "Team", away_team, "OpponentCategory", home_cat, "TeamVsOppCat"),
            "Away_OppCatConcedeVsTeam": get_safe(opp_cat_concede_vs_team, "Team", away_team, "OpponentCategory", home_cat, "OppCatConcedeVsTeam"),
            "Away_CatVsOpp": get_safe(cat_vs_opp, "Category", away_cat, "Opponent", home_team, "CatVsOpp"),
            "Away_OppConcedeVsCat": get_safe(opp_concede_vs_cat, "Team", home_team, "OpponentCategory", away_cat, "OppConcedeVsCat"),
        }])

        probs = {f"{n}+ Goals": f"{round(goal_models[f'Goals_{n}+'].predict_proba(X)[0][1] * 100, 1)}%" for n in range(1, 7)}
        return pd.DataFrame([probs], index=[f"{home_team} vs {away_team}"])

    return match_goal_probabilities
