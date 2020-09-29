"""Loads different apps on different urls."""
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
import budget
import mortgage

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Mortgage", href="/mortgage")),
        dbc.NavItem(dbc.NavLink("Budget", href="/budget")),
    ],
    brand="Mortgage app",
    brand_href="/",
)

# TODO: Store key variables so can store data between pages
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/mortgage':
        return mortgage.layout
    elif pathname == '/budget':
        return budget.layout
    else:
        return '404'


if __name__ == '__main__':
    app.run_server(debug=True)
