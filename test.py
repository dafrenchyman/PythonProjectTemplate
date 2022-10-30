import pandas as pd

# read the dataframe
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
)
# print(df)

# A list of strings containing predictors + A string containing the response

responses = ["traffic_volume"]
predictors = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    "date_time",
]
