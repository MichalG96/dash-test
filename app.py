import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

app = Dash(__name__)

server = app.server

df = pd.read_csv("data.csv", names=["date", "time_of_day", "weight"]).replace(
    {"r": "morning", "w": "evening"}
)

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


app.layout = html.Div(
    children=[
        html.H1(children="Welcome"),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
