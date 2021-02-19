"""Contains the layout and callbacks for the Asset allocation page."""
import datetime
import json
import math
from typing import Optional, Tuple, Dict, Union, List

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import numpy_financial as npf

from app import app

allocation_card = dbc.Card(
    [
        dbc.CardHeader("Asset allocation"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Total wealth (£ ,000)"),
                        dbc.Input(id="total-wealth-allocation", type="number", min=0, step=1, max=100000,),
                    ]
                ),
                dbc.FormGroup([dbc.Label("Mortgage"), dcc.Dropdown(id="mortgage-dropdown")]),
                dbc.FormGroup(
                    [
                        dbc.Label("Property allocation"),
                        dcc.Slider(id="property-allocation", min=0, step=1, value=0,),
                    ]
                ),
                dbc.FormGroup(
                    [dbc.Label("Cash allocation"), dcc.Slider(id="cash-allocation", min=0, step=1, value=0,),]
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

income_card = dbc.Card(
    [
        dbc.CardHeader("Income"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Property (% annual)"),
                        dbc.Input(id="property-r", type="number", min=0, step=0.1, max=1000, value=2.0,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Cash (% annual)"),
                        dbc.Input(id="cash-r", type="number", min=0, step=0.1, max=1000, value=0.5,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Securities (% annual)"),
                        dbc.Input(id="securities-r", type="number", min=0, step=0.1, max=1000, value=4.0,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Rental income (monthly)"),
                        dbc.Input(id="rental-income", type="number", min=0, step=1, max=10000, value=0,),
                    ]

                )
            ]
        ),
    ]
)

one_off_costs_card = dbc.Card(
    [
        dbc.CardHeader("One off costs"),
        dbc.CardBody([
            dbc.FormGroup(
                    [
                        dbc.Label("Mortgage fees"),
                        dbc.Input(id="mortgage-fees-cost", type="number", min=0, step=10, max=10000, value=0,),
                    ]
                ),
            dbc.FormGroup(
                [
                    dbc.Label("Stamp duty"),
                    dbc.Input(id="stamp-duty-cost", type="number", min=0, step=10, max=10000, value=0, ),
                ]
            ),
            ]

        )
    ]
)

monthly_costs_card = dbc.Card(
    [
        dbc.CardHeader("Monthly costs"),
        dbc.CardBody([
            dbc.FormGroup(
                    [
                        dbc.Label("Rent"),
                        dbc.Input(id="rent-cost", type="number", min=0, step=10, max=10000, value=0,),
                    ]
                ),
            dbc.FormGroup(
                [
                    dbc.Label("Housing upkeep"),
                    dbc.Input(id="housing-upkeep-cost", type="number", min=0, step=10, max=10000, value=0, ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Bills"),
                    dbc.Input(id="bills-cost", type="number", min=0, step=10, max=10000, value=0, ),
                ]
            ),
            ]
        )
    ]
)

chart_card = dbc.Card(
    [dbc.CardHeader("Asset value over time"), dbc.CardBody([dcc.Graph(id="allocation-plot")]),]
)

layout = html.Div(
    [
        dbc.Row([dbc.Col(allocation_card, width=6), dbc.Col(income_card, width=6), ]),
        dbc.Row([dbc.Col(one_off_costs_card, width=6), dbc.Col(monthly_costs_card, width=6), ]),
        dbc.Row([dbc.Col(chart_card, width=12)]),
    ]
)


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
def update_sliders(
    total_wealth: Optional[int],
    property_allocation: Optional[int],
    cash_allocation: Optional[int],
    securities_allocation: Optional[int],
) -> Tuple[int, int, int, Dict[int, str], Dict[int, str], Dict[int, str]]:
    """Updates sliders such that the max values add up to total wealth. Updates markers to
    a range of sensible values according to total wealth.

    Updates only if all input values are defined.

    Args:
        total_wealth: total wealth (£ thousands)
        property_allocation: user allocation to property (£ thousands)
        cash_allocation: user allocation to cash (£ thousands)
        securities_allocation: user allocation to securities (£ thousands)
    """
    if all(
        v is not None for v in [total_wealth, property_allocation, cash_allocation, securities_allocation,]
    ):
        # Get the maximum values for each slider
        max_property = total_wealth - (cash_allocation + securities_allocation)
        max_cash = total_wealth - (property_allocation + securities_allocation)
        max_securities = total_wealth - (cash_allocation + property_allocation)

        # Get the step sizes for the marks
        order_of_magnitude = math.floor(math.log10(total_wealth)) - 1
        multiple = int(str(total_wealth)[0]) * 10
        step_size = int(multiple ** order_of_magnitude) if order_of_magnitude >= 0 else 1

        # Get a dictionary of markers in correct format
        property_marks = {i: f"{i: ,}" for i in range(0, max_property + step_size, step_size)}
        cash_marks = {i: f"{i: ,}" for i in range(0, max_cash + step_size, step_size)}
        securities_marks = {i: f"{i: ,}" for i in range(0, max_securities + step_size, step_size)}

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
    Output("total-wealth-allocation", "value"), [Input("url", "pathname")], [State("data-store", "data")],
)
def fill_wealth_value(url: str, data: str) -> int:
    """"
    Returns the total savings value defined by user on budget page, rounded to the nearest thousand.

    Args:
        url: change in url triggers callback
        data: json string

    Returns:
        total wealth rounded to nearest thousand, or zero if not provided.
    """
    if data:
        data = json.loads(data)
        savings = int(round(data.get("savings")))
        return savings
    else:
        return 0


@app.callback(
    Output("allocation-plot", "figure"),
    [
        Input("cash-r", "value"),
        Input("property-r", "value"),
        Input("securities-r", "value"),
        Input("cash-allocation", "value"),
        Input("property-allocation", "value"),
        Input("securities-allocation", "value"),
    ],
)
def update_plot(
    cash_r, property_r, securities_r, cash_allocation, property_allocation, securities_allocation,
):

    fig = go.Figure()

    start_date = datetime.datetime.today()
    end_date = start_date + datetime.timedelta(days=10 * 365)
    periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    x = pd.date_range(datetime.datetime.now(), periods=periods, freq="M")

    cash_array = np.array(
        [npf.fv((cash_r / 100) / 12, i, 0, -(cash_allocation * 1000)) for i in range(periods)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=cash_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "black", "width": 0.8},
            name="Cash",
            hovertemplate="Date: %{x} \nCash value: £%{y:,.3r}<extra></extra>",
        )
    )

    property_array = np.array(
        [npf.fv((property_r / 100) / 12, i, 0, -(property_allocation * 1000)) for i in range(periods)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=property_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "red", "width": 0.8},
            name="Property",
            hovertemplate="Date: %{x} \nProperty value: £%{y:,.3r}<extra></extra>",
        )
    )

    securities_array = np.array(
        [npf.fv((securities_r / 100) / 12, i, 0, -(securities_allocation * 1000)) for i in range(periods)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=securities_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "blue", "width": 0.8},
            name="Securities",
            hovertemplate="Date: %{x} \nSecurities value: £%{y:,.3r}<extra></extra>",
        )
    )

    total_array = cash_array + property_array + securities_array

    fig.add_trace(
        go.Scatter(
            x=x,
            y=total_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "black", "width": 1.5},
            name="Total",
            hovertemplate="Date: %{x} \nTotal value: £%{y:,.3r}<extra></extra>",
        )
    )

    return fig


@app.callback(
    Output("mortgage-dropdown", "options"),
    [Input("url", "pathname")],
    [State("data-store-mortgage", "data")],
)
def fill_dropdown_options(url: str, data: str) -> List[Dict[str, Union[str, int]]]:
    """
    Fills the mortgage dropdown menu to allow users to select from mortgages saved on the mortgage
    page.

    Args:
        url: change in url triggers callback
        data: JSON str of saved mortgages

    Returns:
        dcc.Dropdown options
    """
    if data:
        data = json.loads(data)
        options = [
            {
                "label": f"£{int(i.get('mortgage_size', 0)):,}k {i.get('term')}y, "
                         f"{i.get('offer_term')}y @{i.get('offer_rate')}% "
                         f"then {i.get('interest_rate')}%. £{int(i.get('deposit'))}k deposit",
                "value": val,
            }
            for val, i in enumerate(data)
        ]
        return options
    else:
        raise PreventUpdate


@app.callback(
    Output("property-allocation", "value"),
    [Input("mortgage-dropdown", "value")],
    [State("data-store-mortgage", "data")],
)
def update_property_allocation(dropdown_val: int, data: str) -> int:
    """
    Updates the property allocation to the value of the mortgage deposit selected.

    Args:
        dropdown_val: index of selected mortgage in data
        data: JSON str of saved mortgages

    Returns:
        deposit size allocated to property
    """
    if dropdown_val is not None:
        data = json.loads(data)
        selected_mortgage = data[dropdown_val]
        deposit = int(selected_mortgage.get("deposit"))
        return deposit
    else:
        raise PreventUpdate
