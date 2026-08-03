import pandas as pd
import numpy as np
import holidays

dates = pd.date_range("2019-01-01", "2019-01-10 23:00:00", freq="h")
df = pd.DataFrame({"price": np.random.rand(len(dates))}, index=dates)

# features
df["price_lag_24"] = df["price"].shift(24)
df["price_lag_48"] = df["price"].shift(48)
df["price_lag_168"] = df["price"].shift(168)

df["date"] = df.index.date
daily_stats = df.groupby("date")["price"].agg(["min", "max", "mean"])
daily_stats = daily_stats.shift(1) # shift to yesterday
daily_stats.columns = ["yest_min", "yest_max", "yest_mean"]

df = df.join(daily_stats, on="date")
df["hour"] = df.index.hour
df["dow"] = df.index.dayofweek

dow_dummies = pd.get_dummies(df["dow"], prefix="dow", drop_first=True)
df = pd.concat([df, dow_dummies], axis=1)

us_holidays = holidays.US(years=df.index.year.unique())
df["is_holiday"] = df.index.map(lambda d: int(d in us_holidays))

print(df.head())
print(df.columns)
