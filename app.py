import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="F1 Race Strategy Simulator", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Race Strategy Simulator")
st.caption("Historical-data baseline: estimates finishing position and podium probability. It does not optimize tyre or pit-stop strategy.")

REQUIRED_FILES = ["races.csv", "results.csv", "drivers.csv", "constructors.csv"]

@st.cache_data(show_spinner="Loading and preparing F1 data...")
def prepare_data(file_bytes):
    races = pd.read_csv(file_bytes["races.csv"], na_values=["\\N"])
    results = pd.read_csv(file_bytes["results.csv"], na_values=["\\N"])
    drivers = pd.read_csv(file_bytes["drivers.csv"], na_values=["\\N"])
    constructors = pd.read_csv(file_bytes["constructors.csv"], na_values=["\\N"])

    race_info = races[["raceId", "year", "round", "circuitId", "name", "date"]].rename(columns={"name": "race_name"})
    driver_info = drivers[["driverId", "driverRef", "code", "forename", "surname", "nationality"]].rename(columns={"nationality": "driver_nationality"})
    constructor_info = constructors[["constructorId", "constructorRef", "name", "nationality"]].rename(columns={"name": "constructor_name", "nationality": "constructor_nationality"})

    df = results.merge(race_info, on="raceId", how="left")
    df = df.merge(driver_info, on="driverId", how="left")
    df = df.merge(constructor_info, on="constructorId", how="left")
    df["driver_name"] = (df["forename"].fillna("").astype(str) + " " + df["surname"].fillna("").astype(str)).str.strip()

    numeric_columns = ["year", "round", "grid", "positionOrder", "points", "laps", "milliseconds", "fastestLapSpeed", "fastestLap", "rank"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

    model_df = df.dropna(subset=["positionOrder", "grid", "year", "raceId"]).copy()
    model_df = model_df[model_df["grid"] > 0].copy()
    model_df = model_df.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

    model_df["driver_previous_finish"] = model_df.groupby("driverId")["positionOrder"].transform(lambda s: s.shift(1))
    model_df["driver_avg_finish_before"] = model_df.groupby("driverId")["positionOrder"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["driver_avg_points_before"] = model_df.groupby("driverId")["points"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["team_previous_finish"] = model_df.groupby("constructorId")["positionOrder"].transform(lambda s: s.shift(1))
    model_df["team_avg_finish_before"] = model_df.groupby("constructorId")["positionOrder"].transform(lambda s: s.shift(1).expanding().mean())
    model_df["team_avg_points_before"] = model_df.groupby("constructorId")["points"].transform(lambda s: s.shift(1).expanding().mean())

    features = ["year", "round", "circuitId", "grid", "driverId", "constructorId",
                "driver_previous_finish", "driver_avg_finish_before", "driver_avg_points_before",
                "team_previous_finish", "team_avg_finish_before", "team_avg_points_before"]
    return df, races, model_df, features

uploaded = {}
cols = st.columns(2)
for i, filename in enumerate(REQUIRED_FILES):
    with cols[i % 2]:
        file = st.file_uploader(f"Upload {filename}", type="csv", key=filename)
        if file is not None:
            uploaded[filename] = file.getvalue()

if len(uploaded) != len(REQUIRED_FILES):
    st.info("Upload all four CSV files to start: races.csv, results.csv, drivers.csv, and constructors.csv.")
    st.stop()

try:
    df, races, model_df, feature_columns = prepare_data(uploaded)
except Exception as e:
    st.error(f"Could not read/prepare the files. Check that the four CSVs have the expected columns. Details: {e}")
    st.stop()

years = sorted(model_df["year"].dropna().astype(int).unique())
if len(years) < 2:
    st.error("At least two different seasons are needed to train and test the model.")
    st.stop()

split_index = min(max(1, int(len(years) * 0.8)), len(years) - 1)
test_start_year = years[split_index]
train_data = model_df[model_df["year"] < test_start_year].copy()
test_data = model_df[model_df["year"] >= test_start_year].copy()
if train_data.empty or test_data.empty:
    st.error("Not enough data after the chronological train/test split.")
    st.stop()

@st.cache_resource
def train_models(train_x, train_y, train_podium_y):
    reg = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1))])
    clf = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1))])
    reg.fit(train_x, train_y)
    clf.fit(train_x, train_podium_y)
    return reg, clf

train_podium = train_data["positionOrder"].between(1, 3).astype(int)
regressor, podium_model = train_models(
    train_data[feature_columns], train_data["positionOrder"], train_podium
)

test_pred = regressor.predict(test_data[feature_columns])
test_podium_true = test_data["positionOrder"].between(1, 3).astype(int)
test_podium_pred = podium_model.predict(test_data[feature_columns])
mae = mean_absolute_error(test_data["positionOrder"], test_pred)
rmse = np.sqrt(mean_squared_error(test_data["positionOrder"], test_pred))
accuracy = accuracy_score(test_podium_true, test_podium_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test seasons", f"{int(test_data['year'].min())}–{int(test_data['year'].max())}")
m2.metric("Finishing-position MAE", f"{mae:.2f} places")
m3.metric("Finishing-position RMSE", f"{rmse:.2f} places")
m4.metric("Podium classification accuracy", f"{accuracy:.1%}")

with st.expander("View classification report"):
    report = classification_report(test_podium_true, test_podium_pred, target_names=["Not podium", "Podium"], zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    st.caption("Accuracy can be misleading when podium results are less common. Check podium precision, recall, and F1 too.")

st.divider()
st.subheader("Predict a hypothetical race")
drivers = sorted(df["driver_name"].dropna().loc[lambda s: s.str.strip().ne("")].unique().tolist())
constructors = sorted(df["constructor_name"].dropna().unique().tolist())
if not drivers or not constructors:
    st.error("No driver or constructor names were found in the uploaded data.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    driver_name = st.selectbox("Driver", drivers)
with c2:
    constructor_name = st.selectbox("Constructor / team", constructors)
with c3:
    prediction_year = st.selectbox("Season", list(range(int(max(years)), int(min(years))-1, -1)))

available_rounds = sorted(races.loc[races["year"] == prediction_year, "round"].dropna().astype(int).unique().tolist())
if not available_rounds:
    available_rounds = sorted(model_df["round"].dropna().astype(int).unique().tolist())
round_number = st.selectbox("Round", available_rounds, index=0)
grid_position = st.slider("Hypothetical starting grid position", min_value=1, max_value=20, value=5)

def simulate(driver, team, year, rnd, grid):
    earlier = model_df[(model_df["year"] < year) | ((model_df["year"] == year) & (model_df["round"] < rnd))]
    dh = earlier[earlier["driver_name"].str.casefold() == driver.casefold()]
    th = earlier[earlier["constructor_name"].str.casefold() == team.casefold()]
    race_match = races[(races["year"] == year) & (races["round"] == rnd)]
    circuit_id = race_match["circuitId"].iloc[0] if len(race_match) else np.nan
    row = {
        "year": year, "round": rnd, "circuitId": circuit_id, "grid": grid,
        "driverId": dh["driverId"].iloc[-1] if len(dh) else np.nan,
        "constructorId": th["constructorId"].iloc[-1] if len(th) else np.nan,
        "driver_previous_finish": dh["positionOrder"].iloc[-1] if len(dh) else np.nan,
        "driver_avg_finish_before": dh["positionOrder"].mean() if len(dh) else np.nan,
        "driver_avg_points_before": dh["points"].mean() if len(dh) else np.nan,
        "team_previous_finish": th["positionOrder"].iloc[-1] if len(th) else np.nan,
        "team_avg_finish_before": th["positionOrder"].mean() if len(th) else np.nan,
        "team_avg_points_before": th["points"].mean() if len(th) else np.nan,
    }
    X = pd.DataFrame([row])[feature_columns]
    return float(regressor.predict(X)[0]), float(podium_model.predict_proba(X)[0, 1])

if st.button("Run prediction", type="primary"):
    predicted_position, podium_probability = simulate(driver_name, constructor_name, int(prediction_year), int(round_number), int(grid_position))
    a, b = st.columns(2)
    a.metric("Estimated finishing position", f"{predicted_position:.1f}")
    b.metric("Estimated podium probability", f"{podium_probability:.1%}")
    st.caption("These are model estimates based on historical records, not guarantees. This version does not model tyre wear, pit stops, weather, or safety cars.")

st.subheader("Compare starting-grid scenarios")
if st.button("Compare grid positions"):
    rows = []
    for grid in [1, 3, 5, 10, 15, 20]:
        pos, prob = simulate(driver_name, constructor_name, int(prediction_year), int(round_number), grid)
        rows.append({"Grid position": grid, "Estimated finishing position": round(pos, 1), "Podium probability": f"{prob:.1%}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
