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

brand = "Pland"

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Budget", href="/budget")),
        dbc.NavItem(dbc.NavLink("Mortgage", href="/mortgage")),
        dbc.NavItem(dbc.NavLink("Asset Allocation", href="/asset_allocation")),
    ],
    brand=brand,
    brand_href="/",
)

app.title = brand

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="data-store", storage_type="session"),
        # Store for mortgages saved on the mortgage page, initiated with no mortgage
        dcc.Store(
            id="data-store-mortgage",
            storage_type="session",
            data=json.dumps([{"deposit": 0, "mortgage_size": 0, "purchase_price": 0}]),
        ),
        # Store for saved scenarios on the asset allocation page, initiated with an empty list
        dcc.Store(id="data-store-allocation-scenarios", storage_type="session", data=json.dumps([])),
        navbar,
        html.Div(id="page-content"),
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname: str) -> html.Div:
    """
    Determines which page layout to display.

    Args:
        pathname: url path

    Returns:
        page layout div
    """
    if pathname == "/mortgage":
        return mortgage.layout
    elif pathname == "/budget":
        return budget.layout
    elif pathname == "/asset_allocation":
        return asset_allocation.layout
    else:
        return budget.layout


if __name__ == "__main__":
    app.run_server(debug=True)
