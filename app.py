import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

app = Dash(__name__)

server = app.server

df = pd.read_csv("data.csv", names=["date", "time_of_day", "weight"])

fig = px.line(df, x="date", y="weight", color="time_of_day")

app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash"),
        html.Div(
            children="""
        Dash: A web application framework for your data.
    """
        ),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
