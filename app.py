import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

app = Dash(__name__)

server = app.server

df = pd.read_csv("data.csv").replace({"r": "morning", "w": "evening"}).dropna()

# TODO: calculate the average for each day, plot moving average for the last 4/7 days

fig = px.line(
    df,
    x="date",
    y="weight",
    color="time_of_day",
    line_shape="spline",
    markers=True,
    text="weight",
)
fig.update_traces(textposition="bottom right")
fig.add_vline(
    x="2022-12-13",
    line_width=1,
    line_dash="dash",
    line_color="green",
    # annotation="Diet start",
)

app.layout = html.Div(
    children=[
        html.H1(children="Welcome"),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
