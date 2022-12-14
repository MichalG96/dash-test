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
    return df


def get_start_avg_weight(df: pd.DataFrame) -> float:
    return (
        df[df["date"] == DIET_START]["weight"].iloc[0]
        + df[df["date"] == DIET_START]["weight"].iloc[1]
    ) / 2


def get_trendline(
    start_avg_weight: float, weekly_coefficient: float = 0.5
) -> tuple[np.array, np.array]:
    x_start = DIET_START
    x_end = DIET_START + datetime.timedelta(days=100)
    x = pd.date_range(start=x_start, end=x_end)
    daily_coefficient = weekly_coefficient / 7
    end_avg_weight = start_avg_weight - (daily_coefficient * len(x))
    y = np.linspace(start_avg_weight, end_avg_weight, len(x))
    return x, y


def get_avg_df(original_df: pd.DataFrame) -> pd.DataFrame:
    prev_day, prev_weight = None, None
    avg_daily = []
    for row in original_df.itertuples():
        curr_day = row.date
        curr_weight = row.weight
        if curr_day == prev_day:
            avg_daily.append((row.date, (prev_weight + curr_weight) / 2))
        prev_day = curr_day
        prev_weight = curr_weight

    return pd.DataFrame(avg_daily, columns=["date", "avg_weight"])


df = prepare_data()
start_avg_weight = get_start_avg_weight(df)
trendline_x, trendline_y = get_trendline(start_avg_weight)
avg_df = get_avg_df(df)

fig = px.line(
    df,
    x="date",
    y="weight",
    color="time_of_day",
    line_shape="spline",
    markers=True,
    text="weight",
)
fig.add_trace(
    go.Scatter(
        x=trendline_x, y=trendline_y, mode="lines+markers", name="Expected loss rate"
    )
)
fig.add_trace(
    go.Scatter(
        x=avg_df["date"],
        y=avg_df["avg_weight"],
        mode="lines+markers",
        line_shape="spline",
        name="Average daily weight",
        visible="legendonly",
    ),
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
        df.iloc[0]["date"] - datetime.timedelta(days=1),
        df.iloc[-1]["date"] + datetime.timedelta(days=1),
    ],
    yaxis_range=[min(df["weight"] - 1), max(df["weight"]) + 1],
)
app.layout = html.Div(
    children=[
        html.H1(children="Welcome"),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
