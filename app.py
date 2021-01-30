"""Main entrypoint for app"""
# TODO: Deploy with: https://dash.plotly.com/deployment
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

server = app.server
