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
from budget import stamp_duty_payable

# TODO: Add some tooltips to explain underlying assumptions / data etc.
# TODO: Add option to save scenario and add a page that shows the various scenarios in a data table

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
                dbc.FormGroup([dbc.Label("Property allocation"), html.H5(id="property-allocation"),]),
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
        dbc.CardBody([dbc.FormGroup([dbc.Label("Mortgage"), dcc.Dropdown(id="mortgage-dropdown")]),]),
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
                ),
            ]
        ),
    ]
)


monthly_costs_card = dbc.Card(
    [
        dbc.CardHeader("Monthly costs"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Mortgage fees"),
                        dbc.Input(
                            id="mortgage-fees-cost", type="number", min=0, step=10, max=10000, value=0,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Stamp duty"),
                        dbc.Input(id="stamp-duty-cost", type="number", min=0, step=100, value=0,),
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
                        dbc.Input(
                            id="housing-upkeep-cost", type="number", min=0, step=10, max=10000, value=0,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Bills (monthly)"),
                        dbc.Input(id="bills-cost", type="number", min=0, step=10, max=10000, value=0,),
                    ]
                ),
            ]
        ),
    ]
)

chart_card = dbc.Card(
    [dbc.CardHeader("Asset value over time"), dbc.CardBody([dcc.Graph(id="allocation-plot")]),]
)

layout = html.Div(
    [
        dbc.Row([dbc.Col(asset_card, width=6), dbc.Col(liability_card, width=6),]),
        dbc.Row([dbc.Col(income_card, width=6), dbc.Col(monthly_costs_card, width=6),]),
        dbc.Row([dbc.Col(chart_card, width=12)]),
    ]
)


@app.callback(
    [
        Output("cash-allocation", "max"),
        Output("securities-allocation", "max"),
        Output("cash-allocation", "marks"),
        Output("securities-allocation", "marks"),
    ],
    [
        Input("total-wealth-allocation", "value"),
        Input("cash-allocation", "value"),
        Input("securities-allocation", "value"),
        Input("stamp-duty-cost", "value"),
        Input("mortgage-fees-cost", "value"),
        Input("mortgage-dropdown", "value"),
    ],
    [State("data-store-mortgage", "data"),],
)
def update_sliders(
    total_wealth: Optional[int],
    cash_allocation: Optional[int],
    securities_allocation: Optional[int],
    stamp_duty: int,
    mortgage_fees: int,
    mortgage_idx: int,
    mortgage_data: str,
) -> Tuple[int, int, Dict[int, str], Dict[int, str]]:
    """Updates sliders such that the max values add up to total wealth. Updates markers to
    a range of sensible values according to total wealth.

    Updates only if all input values are defined.

    Args:
        mortgage_data: JSON str of saved mortgages
        mortgage_idx: index of selected mortgage
        mortgage_fees: total mortgage fees (£)
        stamp_duty: amount of stamp duty payable for selected mortgage  (£)
        total_wealth: total wealth (£ thousands)
        cash_allocation: user allocation to cash (£ thousands)
        securities_allocation: user allocation to securities (£ thousands)
    """
    if all(v is not None for v in [total_wealth, cash_allocation, securities_allocation, mortgage_idx]):
        mortgage_data = json.loads(mortgage_data)
        mortgage = mortgage_data[mortgage_idx]
        deposit = int(mortgage.get("deposit"))
        mortgage_fees = int(mortgage_fees / 1000)
        stamp_duty = int(stamp_duty / 1000)

        # Get the maximum values for each slider
        max_cash = total_wealth - (deposit + securities_allocation + mortgage_fees + stamp_duty)
        max_securities = total_wealth - (cash_allocation + deposit + mortgage_fees + stamp_duty)

        # Get the step sizes for the marks
        order_of_magnitude = math.floor(math.log10(max_cash + max_securities)) - 1
        multiple = int(str(total_wealth)[0]) * 10
        step_size = int(multiple ** order_of_magnitude) if order_of_magnitude >= 0 else 1

        # Get a dictionary of markers in correct format
        cash_marks = {i: f"{i: ,}" for i in range(0, max_cash + step_size, step_size)}
        securities_marks = {i: f"{i: ,}" for i in range(0, max_securities + step_size, step_size)}

        return (
            max_cash,
            max_securities,
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
    cash_r: float,
    property_r: float,
    securities_r: float,
    cash_allocation: int,
    securities_allocation: int,
    mortgage_idx: int,
    stamp_duty: int,
    mortgage_data: str,
) -> go.Figure:
    """
    Updates plot showing household balance sheet and income / expenditure over time.

    Args:
        cash_r: rate of return on cash (%)
        property_r: rate of return on property (%)
        securities_r: rate of return on securities (%)
        cash_allocation: user initial allocation to cash (£ ,000)
        securities_allocation: user initial allocation to securities (£ ,000)
        mortgage_idx: index of selected mortgage
        stamp_duty: stamp duty payable (£)
        mortgage_data: JSON str of saved mortgage data

    Returns:
        go.Figure object of household balance sheet and income / expenditure statement
    """
    # Will only show plot if a mortgage has been selected
    if mortgage_idx is None:
        raise PreventUpdate

    data = json.loads(mortgage_data)
    mortgage = data[mortgage_idx]

    # Set date range to mortgage term
    term = mortgage.get("term") * 12
    x = pd.date_range(datetime.datetime.now(), periods=term, freq="M")

    fig = go.Figure()

    # 1. Income

    # TODO: Show bar chart of expected rental income / income savings

    # 2. Expenditure

    # TODO: Add mortgage fees and ongoing expenses to Bar trace

    # Stamp duty one off cost in period one
    stamp_duty_exp = np.append(np.array([-stamp_duty]), np.zeros(len(x) - 1))

    fig.add_trace(go.Bar(x=x, y=stamp_duty_exp, name="Stamp duty"))

    # 3. Liabilities
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

    # 4. Assets
    cash_array = np.array([npf.fv((cash_r / 100) / 12, i, 0, -(cash_allocation * 1000)) for i in range(term)])

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

    # TODO: Assume all saved income reinvested in securities (update pmt arg (0) to payment)
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
    [Output("property-allocation", "children"), Output("stamp-duty-cost", "value"),],
    [Input("mortgage-dropdown", "value")],
    [State("data-store-mortgage", "data"), State("data-store", "data"),],
)
def update_property_allocation(dropdown_val: int, mortgage_data: str, data: str) -> Tuple[str, float]:
    """
    Updates the property allocation to the value of the mortgage deposit selected.

    Args:
        dropdown_val: index of selected mortgage in data
        mortgage_data: JSON str of saved mortgages
        data: JSON str of data from budget page

    Returns:
        deposit size allocated to property
    """
    if dropdown_val is not None:
        mortgage_data = json.loads(mortgage_data)
        selected_mortgage = mortgage_data[dropdown_val]
        deposit = f"£{int(selected_mortgage.get('deposit')) * 1000: ,}"

        # Get the amount of stamp duty payable
        purchase_price = selected_mortgage.get("purchase_price") * 1000
        data = json.loads(data)
        higher_rate = True if data.get("stamp_duty_rate") == "higher_rate" else False
        sdp = int(stamp_duty_payable(purchase_price, higher_rate))

        return deposit, sdp
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
