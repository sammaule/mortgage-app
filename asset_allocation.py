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

# TODO: Update to be consistent with household balance sheet on paper

asset_card = dbc.Card(
    [
        dbc.CardHeader("Assets"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Total wealth (£ ,000)"),
                        dbc.Input(id="total-wealth-allocation", type="number", min=0, step=1, max=100000,),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Property allocation"),
                        # TODO: update so just text updated on selection of mortgage
                        dcc.Slider(id="property-allocation", min=0, step=1, value=0, ),
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

liability_card = dbc.Card(
    [
        dbc.CardHeader("Liabilities"),
        dbc.CardBody(
            [
                dbc.FormGroup([dbc.Label("Mortgage"), dcc.Dropdown(id="mortgage-dropdown")]),

            ]
        )
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


monthly_costs_card = dbc.Card(
    [
        dbc.CardHeader("Monthly costs"),
        dbc.CardBody([
            dbc.FormGroup(
                [
                    dbc.Label("Mortgage fees"),
                    dbc.Input(id="mortgage-fees-cost", type="number", min=0, step=10, max=10000, value=0, ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Stamp duty"),
                    dbc.Input(id="stamp-duty-cost", type="number", min=0, step=100, value=0, ),
                ]
            ),
            dbc.FormGroup(
                    [
                        dbc.Label("Rent (monthly)"),
                        dbc.Input(id="rent-cost", type="number", min=0, step=10, max=10000, value=0,),
                    ]
                ),
            dbc.FormGroup(
                [
                    dbc.Label("Housing upkeep (monthly)"),
                    dbc.Input(id="housing-upkeep-cost", type="number", min=0, step=10, max=10000, value=0, ),
                ]
            ),
            dbc.FormGroup(
                [
                    dbc.Label("Bills (monthly)"),
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
        dbc.Row([dbc.Col(asset_card, width=6), dbc.Col(liability_card, width=6), ]),
        dbc.Row([dbc.Col(income_card, width=6), dbc.Col(monthly_costs_card, width=6), ]),
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
        Input("securities-allocation", "value"),
        Input("mortgage-dropdown", "value"),
        Input("stamp-duty-cost", "value"),
    ],
    [State("data-store-mortgage", "data")],
)
def update_plot(
    cash_r, property_r, securities_r, cash_allocation, securities_allocation, mortgage_idx,
        stamp_duty, mortgage_data
):
    # Will only show plot if a mortgage has been selected
    if mortgage_idx is None:
        raise PreventUpdate

    data = json.loads(mortgage_data)
    mortgage = data[mortgage_idx]

    # Set date range to mortgage term
    term = mortgage.get("term") * 12
    x = pd.date_range(datetime.datetime.now(), periods=term, freq="M")

    fig = go.Figure()

    # 1. Liabilities
    mortgage_balance = _get_mortgage_balance(mortgage)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mortgage_balance,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "red", "width": 1.5},
            name="Liabilities",
            hovertemplate="Date: %{x} \nTotal value: £%{y:,.3r}<extra></extra>",
        )
    )

    # 2. Assets

    cash_array = np.array(
        [npf.fv((cash_r / 100) / 12, i, 0, -(cash_allocation * 1000)) for i in range(term)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=cash_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "green", "width": 0.8},
            name="Cash",
            hovertemplate="Date: %{x} \nCash value: £%{y:,.3r}<extra></extra>",
        )
    )

    purchase_price = mortgage.get("purchase_price")
    property_array = np.array(
        [npf.fv((property_r / 100) / 12, i, 0, -(purchase_price * 1000)) for i in range(term)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=property_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "orange", "width": 0.8},
            name="Property",
            hovertemplate="Date: %{x} \nProperty value: £%{y:,.3r}<extra></extra>",
        )
    )

    securities_array = np.array(
        [npf.fv((securities_r / 100) / 12, i, 0, -(securities_allocation * 1000)) for i in range(term)]
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=securities_array,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "purple", "width": 0.8},
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
            line={"color": "green", "width": 1.5},
            name="Total assets",
            hovertemplate="Date: %{x} \nTotal value: £%{y:,.3r}<extra></extra>",
        )
    )

    # 3. Income

    # 4. Expenditure

    # Stamp duty one off cost in period one
    stamp_duty_exp = np.append(np.array([-stamp_duty]), np.zeros(len(x) - 1))

    fig.add_trace(
        go.Bar(
            x=x,
            y=stamp_duty_exp,
            name="Stamp duty"
        )
    )
    # 5. Wealth

    wealth = total_array - mortgage_balance

    fig.add_trace(
        go.Scatter(
            x=x,
            y=wealth,
            fillcolor="rgba(0,176,246,0.2)",
            line={"color": "black", "width": 2},
            name="Wealth",
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


# TODO: Tidy function, add docstring etc.
def _get_mortgage_balance(mortgage):

    mortgage_size = int(mortgage.get("mortgage_size")) * 1000
    term = mortgage.get("term") * 12
    offer_term = mortgage.get("offer_term") * 12

    remaining_term = term - offer_term

    interest_rate = (mortgage.get("interest_rate") / 12) / 100
    offer_rate = (mortgage.get("offer_rate") / 12) / 100  # monthly interest rate

    per = np.arange(term) + 1
    offer_principal_payments = (-1 * npf.ppmt(offer_rate, per, term, mortgage_size))[:offer_term]

    balance_after_offer = mortgage_size - np.sum(offer_principal_payments)

    per = np.arange(remaining_term) + 1
    remaining_principal_payments = -1 * npf.ppmt(interest_rate, per, remaining_term, balance_after_offer)

    principal_payments = np.append(offer_principal_payments, remaining_principal_payments)

    outstanding_balance = mortgage_size - np.cumsum(principal_payments)

    return outstanding_balance