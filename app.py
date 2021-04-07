"""Main entrypoint for app"""
import dash
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

server = app.server
