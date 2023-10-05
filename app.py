import datetime
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html
from plotly.subplots import make_subplots

app = Dash(__name__)

server = app.server

DIET_START = datetime.date(2022, 12, 13)


def prepare_data() -> pd.DataFrame:
    df_parquet = pd.read_parquet("data/data.parquet").dropna()
    df_current = pd.read_csv("data_current.csv").dropna()
    df = pd.concat([df_parquet, df_current], ignore_index=True)
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
    x_end = DIET_START + datetime.timedelta(days=400)
    x = pd.date_range(start=x_start, end=x_end)
    daily_coefficient = weekly_coefficient / 7
    end_avg_weight = start_avg_weight - (daily_coefficient * len(x))
    y = np.clip(
        np.round(np.squeeze(np.linspace(start_avg_weight, end_avg_weight, len(x))), 2),
        88.0,
        None,
    )
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

bar_df = pd.DataFrame({"value": trendline_y}, index=trendline_x).join(
    df["moving_average"]
)
bar_df["diff"] = bar_df["value"] - bar_df["moving_average"]
bar_df["signum"] = np.where(bar_df["diff"] > 0, "forestgreen", "crimson")

fig_line = px.line(
    df,
    markers=True,
    line_shape="spline",
    # For some reason makes the trace disappear
    # hover_data={'weight_evening': ':.2f'}
)
fig_line.update_traces(connectgaps=True, textposition="bottom right", marker_size=5)


# TODO:
#   Adds text for each trace. Problem: text doesn't disappear when trace is hidden
# add_weight_text(df, fig)
def rand_func():
    if random.random() > 0.6:
        return "green"
    return "red"


fig_line.add_trace(
    go.Scatter(
        x=trendline_x,
        y=trendline_y,
        mode="lines",
        name="Expected loss rate",
        fill="tonexty",
        fillcolor="rgba(0,0,0, 0.2)",
        # line_shape="spline",
        # line_smoothing=1.3
        # fillcolor=rand_func()
    )
)

fig_line.update_traces(textposition="bottom right")

fig_bar = go.Figure(
    go.Bar(name="Diff", y=bar_df["diff"], x=bar_df.index, marker_color=bar_df["signum"])
)
fig_bar.update_layout(barmode="stack")

figures = [fig_line, fig_bar]
fig = make_subplots(
    rows=len(figures),
    cols=1,
    row_heights=[0.8, 0.2],
    shared_xaxes=True,
    vertical_spacing=0.01,
)

for i, figure in enumerate(figures):
    for trace in range(len(figure["data"])):
        fig.add_trace(figure["data"][trace], row=i + 1, col=1)

fig.update_layout(
    xaxis=dict(
        range=[
            df.index[0] - datetime.timedelta(days=1),
            df.index[-1] + datetime.timedelta(days=1),
        ],
        dtick=86400000.0,
    ),
    yaxis=dict(
        range=[min(df["weight_morning"] - 0.5), max(df["weight_evening"]) + 0.3],
        dtick=0.5,
        tick0=90.0,
        ticklabelstep=2,
    ),
    height=750,  # in px
)
fig.add_vline(
    x=DIET_START,
    line_width=1,
    line_dash="dash",
    line_color="green",
    # annotation="Diet start",
)
app.layout = html.Div(
    children=[
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
