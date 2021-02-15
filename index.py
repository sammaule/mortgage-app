"""Loads different apps on different urls."""
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
import budget
import mortgage
import asset_allocation

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Budget", href="/budget")),
        dbc.NavItem(dbc.NavLink("Mortgage", href="/mortgage")),
        dbc.NavItem(dbc.NavLink("Asset Allocation", href="/asset_allocation"))
    ],
    brand="Mortgage app",
    brand_href="/",
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="data-store", storage_type="session"),
    dcc.Store(id="data-store-mortgage", storage_type="session"),
    navbar,
    # TODO: Add welcome to app page - with "next" button to take user to budget page
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/mortgage':
        return mortgage.layout
    elif pathname == '/budget':
        return budget.layout
    elif pathname == '/asset_allocation':
        return asset_allocation.layout
    else:
        return budget.layout


if __name__ == '__main__':
    app.run_server(debug=True)
