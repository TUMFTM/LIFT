#!/usr/bin/env python3


def map_timeframes(df, key):
    cs_map = {"bev": map_timeframes_vehicles, "icev": map_timeframes_vehicles, "mb": map_timeframes_batteries}
    return cs_map[key](df)


def map_timeframes_vehicles(df):
    condition = df.index.weekday > 4

    df.loc[condition, "timeframe"] = "weekend"
    df.loc[condition, "demand_mean"] = 10
    df.loc[condition, "demand_std"] = 5

    df.loc[~condition, "timeframe"] = "weekday"
    df.loc[~condition, "demand_mean"] = 40
    df.loc[~condition, "demand_std"] = 10

    return df["timeframe"], df["demand_mean"], df["demand_std"]


def map_timeframes_batteries(df):
    df.loc[:, "timeframe"] = "day"
    df.loc[:, "demand_mean"] = 5
    df.loc[:, "demand_std"] = 2

    return df["timeframe"], df["demand_mean"], df["demand_std"]
