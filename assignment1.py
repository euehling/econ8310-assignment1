import pandas as pd
import numpy as np
from pygam import LinearGAM, s


train = pd.read_csv("assignment_data_train.csv")
train.columns = train.columns.str.lower()
train["timestamp"] = pd.to_datetime(train["timestamp"])


y = train["trips"].values
X = train[["hour", "day", "month"]].values


model = LinearGAM(
    s(0, n_splines=24) +   # hour of day
    s(1, n_splines=10) +   # day of month
    s(2, n_splines=12)     # month
)

modelFit = model.fit(X, y)


test = pd.read_csv("assignment_data_test.csv")
test.columns = test.columns.str.lower()
test["timestamp"] = pd.to_datetime(test["timestamp"])


X_test = test[["hour", "day", "month"]].values


pred = modelFit.predict(X_test)


# print(len(pred))  # should be 744
# print(pred[:10])
