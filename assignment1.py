import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
import matplotlib.pyplot as plt
from pygam import LinearGAM, s

# ===============================
# EXPONENTIAL SMOOTHING
# ===============================
df = pd.read_csv("assignment_data_train.csv")
print(df.info())


# timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values(df.columns[0]).set_index(df.columns[0])

# trips series (hourly)
y = df["trips"].astype(float).asfreq("h")

if y.isna().any():
    y = y.interpolate(limit_direction="both")

model = ES(
    y,
    trend="add",
    seasonal="add",
    seasonal_periods=24
)

modelFit = model.fit()

plt.figure(figsize=(12, 5))
plt.plot(y, label="Actual")
plt.plot(modelFit.fittedvalues, label="Fitted", alpha=0.8)
plt.legend()
plt.title("Exponential Smoothing Fit")
plt.xlabel("Time")
plt.ylabel("Taxi Trips")
plt.show()

# ===============================
# GAM
# ===============================

train = pd.read_csv("assignment_data_train.csv")
train.columns = train.columns.str.lower()
train["timestamp"] = pd.to_datetime(train["timestamp"])

y_gam = train["trips"].values
X = train[["hour", "day", "month"]].values

model = LinearGAM(
    s(0, n_splines=24) +
    s(1, n_splines=10) +
    s(2, n_splines=12)
)

modelFit = model.fit(X, y_gam)

# ===============================
# GAM PLOTS
# ===============================

# ----- Hour of day -----
hours = np.arange(0, 24)
X_hour = np.zeros((24, 3))
X_hour[:, 0] = hours
X_hour[:, 1] = train["day"].median()
X_hour[:, 2] = train["month"].median()

pd_hour = modelFit.partial_dependence(term=0, X=X_hour)

plt.figure()
plt.plot(hours, pd_hour)
plt.xlabel("Hour of Day")
plt.ylabel("Effect on Trips")
plt.title("GAM Hour of Day Effect")
plt.show(block=True)

# ----- Day of month -----
days = np.arange(1, 32)
X_day = np.zeros((31, 3))
X_day[:, 0] = train["hour"].median()
X_day[:, 1] = days
X_day[:, 2] = train["month"].median()

pd_day = modelFit.partial_dependence(term=1, X=X_day)

plt.figure()
plt.plot(days, pd_day)
plt.xlabel("Day of Month")
plt.ylabel("Effect on Trips")
plt.title("GAM Day of Month Effect")
plt.show(block=True)

# ----- Month -----
months = np.arange(1, 13)
X_month = np.zeros((12, 3))
X_month[:, 0] = train["hour"].median()
X_month[:, 1] = train["day"].median()
X_month[:, 2] = months

pd_month = modelFit.partial_dependence(term=2, X=X_month)

plt.figure()
plt.plot(months, pd_month)
plt.xlabel("Month")
plt.ylabel("Effect on Trips")
plt.title("GAM Month Effect")
plt.show(block=True)