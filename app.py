import datetime
import random

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
    y = np.round(np.squeeze(np.linspace(start_avg_weight, end_avg_weight, len(x))), 2)
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


# Alternative way for moving average.
# Later discovered pandas' df.rolling(window_size).mean()
# def moving_average(arr, window_size: int = 7):
#     result = np.convolve(arr, np.ones(window_size), "valid") / window_size
#     result = np.pad(
#         result, (len(arr) - len(result), 0), "constant", constant_values=(np.nan,)
#     )
#     return result


def add_weight_text(df, fig):
    col_name_y_mapping = {
        "weight_morning": -15,
        "weight_evening": 15,
        "avg_weight": -10,
    }

    def add_text_for_column(col_name):
        if not np.isnan(getattr(row, col_name)):
            fig.add_annotation(
                x=row.Index,
                y=getattr(row, col_name),
                text=f"{getattr(row, col_name)}",
                showarrow=False,
                yshift=col_name_y_mapping[col_name],
                xshift=10,
            )

    for row in df.itertuples():
        for col in col_name_y_mapping:
            add_text_for_column(col)


df = prepare_data()
df = get_avg_df(df)
trendline_x, trendline_y = get_trendline(df)
# df["moving_average"] = moving_average(df["avg_weight"])
df["moving_average"] = df["avg_weight"].rolling(7).mean()
df = df.round(2)
# print(df)
# TODO: add legend, labels and axis names

fig = px.line(
    df,
    markers=True,
    line_shape="spline",
    height=700,  # in px
    # For some reason makes the trace disappear
    # hover_data={'weight_evening': ':.2f'}
)
fig.update_traces(connectgaps=True, textposition="bottom right", marker_size=5)

# TODO:
#   Adds text for each trace. Problem: text doesn't disappear when trace is hidden
# add_weight_text(df, fig)
def rand_func():
    if random.random() > 0.6:
        return "green"
    return "red"


fig.add_trace(
    go.Scatter(
        x=trendline_x,
        y=trendline_y,
        mode="lines",
        name="Expected loss rate",
        fill="tonexty",
        fillcolor="rgba(0,0,0, 0.2)"
        # fillcolor=rand_func()
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
    yaxis_range=[min(df["weight_morning"] - 0.5), max(df["weight_evening"]) + 0.5],
)
app.layout = html.Div(
    children=[
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
