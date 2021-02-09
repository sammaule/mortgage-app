"""Contains the layout and callbacks for the Asset allocation page."""
import json
import math

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State

from dash.exceptions import PreventUpdate


from app import app

# TODO: Add boxes for user to fill out the cost / return profiles of each type of investment.
#       a) housing costs: mortgage, one off (tax etc.), upkeep, bills
#          housing returns: house price growth, rent
#       b) cash costs: bank fees?
#          cash returns: interest rate
#       c) securities costs: fees
#          securities returns: expected annual returns

allocation_card = dbc.Card(
    [
        dbc.CardHeader("Asset allocation"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Total wealth (£ ,000)"),
                        dbc.Input(
                            id="total-wealth-allocation",
                            type="number",
                            min=0,
                            max=100000,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Property allocation"),
                        dcc.Slider(id="property-allocation", min=0, step=1, value=0,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Cash allocation"),
                        dcc.Slider(id="cash-allocation", min=0, step=1, value=0,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Securities allocation"),
                        dcc.Slider(id="securities-allocation", min=0, step=1, value=0,),
                    ]
                ),
            ]
        ),
    ]
)

layout = html.Div(dbc.Row([dbc.Col(allocation_card, width=6),]))


@app.callback(
    [
        Output("property-allocation", "max"),
        Output("cash-allocation", "max"),
        Output("securities-allocation", "max"),
        Output("property-allocation", "marks"),
        Output("cash-allocation", "marks"),
        Output("securities-allocation", "marks"),
    ],
    [
        Input("total-wealth-allocation", "value"),
        Input("property-allocation", "value"),
        Input("cash-allocation", "value"),
        Input("securities-allocation", "value"),
    ],
)
def fill_data_values(
    total_wealth, property_allocation, cash_allocation, securities_allocation
):
    if all(
        v is not None
        for v in [
            total_wealth,
            property_allocation,
            cash_allocation,
            securities_allocation,
        ]
    ):
        # Get the maximum values for each slider
        max_property = total_wealth - (cash_allocation + securities_allocation)
        max_cash = total_wealth - (property_allocation + securities_allocation)
        max_securities = total_wealth - (cash_allocation + property_allocation)

        # Get the step sizes for the marks
        order_of_magnitude = math.floor(math.log10(total_wealth)) - 1
        multiple = int(str(total_wealth)[0]) * 10
        step_size = (
            int(multiple ** order_of_magnitude) if order_of_magnitude >= 0 else 1
        )

        # Get a dictionary of markers in correct format
        property_marks = {
            i: f"{i: ,}" for i in range(0, max_property + step_size, step_size)
        }
        cash_marks = {i: f"{i: ,}" for i in range(0, max_cash + step_size, step_size)}
        securities_marks = {
            i: f"{i: ,}" for i in range(0, max_securities + step_size, step_size)
        }

        return (
            max_property,
            max_cash,
            max_securities,
            property_marks,
            cash_marks,
            securities_marks,
        )
    else:
        raise PreventUpdate


@app.callback(
    Output("total-wealth-allocation", "value"),
    [Input("url", "pathname")],
    [State("data-store", "data")],
)
def fill_data_values(url, data):
    """

    Args:
        data (str): json string

    Returns:
        deposit size, purchase price and income (Tuple[float, float, float])
    """
    if data:
        data = json.loads(data)
        savings = round(data.get("savings"), 3)
        return savings
