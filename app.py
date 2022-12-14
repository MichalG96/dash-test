import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

app = Dash(__name__)

server = app.server

DIET_START = datetime.date(2022, 12, 13)


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv").replace({"r": "morning", "w": "evening"}).dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df_morning = df[df["time_of_day"] == "morning"]
    df_evening = df[df["time_of_day"] == "evening"]
    df = (
        df_morning.set_index("date")
        .join(
            df_evening.set_index("date"),
            lsuffix="_morning",
            rsuffix="_evening",
            how="outer",
        )
        .drop(columns=["time_of_day_morning", "time_of_day_evening"])
    )
    return df


def get_start_avg_weight(df: pd.DataFrame) -> float:
    return df[df.index == DIET_START]["avg_weight"]


def get_trendline(df: pd.DataFrame, weekly_coefficient: float = 0.5) -> tuple:
    start_avg_weight = get_start_avg_weight(df)
    x_start = DIET_START
    x_end = DIET_START + datetime.timedelta(days=100)
    x = pd.date_range(start=x_start, end=x_end)
    daily_coefficient = weekly_coefficient / 7
    end_avg_weight = start_avg_weight - (daily_coefficient * len(x))
    y = np.squeeze(np.linspace(start_avg_weight, end_avg_weight, len(x)))
    return x, y


def get_avg_df(original_df: pd.DataFrame) -> pd.DataFrame:
    def count_avg(row, coeff):
        if not (np.isnan(row["weight_morning"]) or np.isnan(row["weight_evening"])):
            return (row["weight_morning"] + row["weight_evening"]) / 2
        if np.isnan(row["weight_evening"]):
            return row["weight_morning"] + coeff
        if np.isnan(row["weight_morning"]):
            return row["weight_evening"] - coeff
        return None

    avg_diff_throughout_a_day = np.mean(
        original_df["weight_evening"] - original_df["weight_morning"]
    )
    original_df["avg_weight"] = original_df.apply(
        count_avg, axis=1, coeff=avg_diff_throughout_a_day / 2
    )
    return original_df


def moving_average(arr, window_size: int = 7):
    result = np.convolve(arr, np.ones(window_size), "valid") / window_size
    result = np.pad(
        result, (len(arr) - len(result), 0), "constant", constant_values=(np.nan,)
    )
    return result


df = prepare_data()
df = get_avg_df(df)
trendline_x, trendline_y = get_trendline(df)
df["moving_average"] = moving_average(df["avg_weight"])

# TODO: add legend, labels and axis names

fig = px.line(
    df,
    markers=True,
    line_shape="spline",
)
fig.update_traces(connectgaps=True)

fig.add_trace(
    go.Scatter(
        x=trendline_x, y=trendline_y, mode="lines+markers", name="Expected loss rate"
    )
)

fig.update_traces(textposition="bottom right")
fig.add_vline(
    x=DIET_START,
    line_width=1,
    line_dash="dash",
    line_color="green",
    # annotation="Diet start",
)
fig.update_layout(
    xaxis_range=[
        df.index[0] - datetime.timedelta(days=1),
        df.index[-1] + datetime.timedelta(days=1),
    ],
    yaxis_range=[min(df["weight_morning"] - 1), max(df["weight_evening"]) + 1],
)
app.layout = html.Div(
    children=[
        html.H1(children="Welcome"),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
