"""Loads different apps on different urls."""
import json

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
import budget
import mortgage
import asset_allocation

# TODO: Update icon

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Budget", href="/budget")),
        dbc.NavItem(dbc.NavLink("Mortgage", href="/mortgage")),
        dbc.NavItem(dbc.NavLink("Asset Allocation", href="/asset_allocation"))
    ],
    brand="Pland",
    brand_href="/",
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="data-store", storage_type="session"),
    dcc.Store(id="data-store-mortgage", storage_type="session",
              data=json.dumps([{
            "deposit": 0,
            "mortgage_size": 0,
            "purchase_price": 0
              }])
              ),
    dcc.Store(id="allocation-scenarios", storage_type="session"),
    navbar,
    # TODO: Add welcome to app page - with "next" button to take user to budget page
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname: str) -> html.Div:
    """
    Determines which page layout to display.

    Args:
        pathname: url path

    Returns:
        page layout div
    """
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
