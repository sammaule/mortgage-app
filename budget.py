"""Code for the budget page of the app."""
import datetime
import json
import re
from datetime import datetime as dt

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app import app
from typing import Tuple

savings_card = dbc.Card(
    [
        dbc.CardHeader("Savings"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Total current savings (£k)"),
                        dbc.Input(id="current-savings", value=180, type="number"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Savings per month (£)"),
                        dbc.Input(id="saving-rate", value=700, step=10, type="number",),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Savings interest (%)"),
                        dbc.Input(id="savings-interest", value=1.12, step=0.01, type="number",),
                    ]
                ),
            ]
        ),
    ]
)

# TODO: Amend this card title and add a dropdown for the target purchase date
income_and_tax_card = dbc.Card(
    [
        dbc.CardHeader("Income, Stamp Duty and Purchase date"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        # TODO: Store income in memory for mortgage page
                        dbc.Label("Income (£k)"),
                        dbc.Input(id="income", value=51, type="number"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Stamp Duty rate"),
                        dbc.Select(
                            id="stamp-duty-rate",
                            options=[
                                {"label": "Normal rate", "value": "normal_rate"},
                                {"label": "Higher rate", "value": "higher_rate"},
                            ],
                            value="higher_rate",
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Target purchase date"),
                        html.Br(),
                        html.Br(),
                        dcc.DatePickerSingle(
                            id="target-purchase-date",
                            date=datetime.datetime.today(),
                            min_date_allowed=datetime.datetime.today(),
                            max_date_allowed=dt(2022, 12, 31),
                            display_format="DD/MM/YYYY",
                        ),
                    ]
                ),
            ]
        ),
    ]
)

budget_results_card = dbc.Card(
    [
        dbc.CardHeader("Budget results"),
        dbc.CardBody(
            [
                html.Div([dbc.Label("Deposit on target date"), html.H5(id="deposit-on-target-date")]),
                html.Div(
                    [dbc.Label("Mortgage available on target date"), html.H5(id="mortgage-on-target-date")]
                ),
                html.Div(
                    [
                        dbc.Label("Total property value:"),
                        html.H5(id="property-value-on-target-date"),
                    ]
                ),
                html.Div([dbc.Label("Stamp duty range:"), html.H5(id="stamp-duty-on-target-date")]),
            ]
        ),
    ]
)

layout = html.Div(
    [
        # TODO: Add a row at the top which asks user to fill in their details
        dbc.Row([dbc.Col([savings_card], width=6), dbc.Col([income_and_tax_card], width=6)]),
        dbc.Row(
            [
                dbc.Col([budget_results_card], width=3),
                dbc.Col(
                    [
                        dbc.Card(
                            [dbc.CardHeader("Maximum budget"), dbc.CardBody(dcc.Graph(id="budget-plot")),]
                        )
                    ],
                    width=9,
                ),
            ]
        ),
    ]
)


@app.callback(
    [
        Output("budget-plot", "figure"),
        Output("deposit-on-target-date", "children"),
        Output("mortgage-on-target-date", "children"),
        Output("property-value-on-target-date", "children"),
        Output("stamp-duty-on-target-date", "children"),
        Output("data-store", "data"),
    ],
    [
        Input("current-savings", "value"),
        Input("saving-rate", "value"),
        Input("savings-interest", "value"),
        Input("income", "value"),
        Input("stamp-duty-rate", "value"),
        Input("target-purchase-date", "date"),
    ],
)
def calc_ltv(
    savings: int, saving_rate: int, r: float, income: int, stamp_duty_rate: str, target_date: str
) -> Tuple[go.Figure, str, str, str, str, str]:
    """
    Callback to populate data in the savings plot according to the input values entered by user.
    Assumes that all interest is paid monthly at a rate of 1/12*r and all reinvested.

    Args:
        savings (int): Total current savings
        saving_rate (int): Amount saved each month
        r (float): interest rate (%)
    """
    # TODO: Improve hover text diplayed to show savings at each timepoint
    if all(v is not None for v in [savings, saving_rate, r]):
        fig = go.Figure()

        x = pd.date_range(datetime.datetime.now(), periods=24, freq="M")

        savings_array = np.array(
            [npf.fv((r / 100) / 12, i, -saving_rate, -(savings * 1000)) for i in range(24)]
        )

        mortgage_h = 4.7 * (income * 1000)
        mortgage_m = 4.5 * (income * 1000)
        mortgage_l = 4.3 * (income * 1000)

        higher_rate = True if stamp_duty_rate == "higher_rate" else False

        budget_h = np.array(
            [iterative_p(s, mortgage_h, stamp_duty_payable, higher_rate)[1] for s in savings_array]
        )
        budget_m = np.array(
            [iterative_p(s, mortgage_m, stamp_duty_payable, higher_rate)[1] for s in savings_array]
        )
        budget_l = np.array(
            [iterative_p(s, mortgage_l, stamp_duty_payable, higher_rate)[1] for s in savings_array]
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=budget_h,
                fillcolor="rgba(0,176,246,0.2)",
                line={"color": "black", "width": 0.8},
                name="4.7x LTI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=budget_m,
                fill="tonexty",
                fillcolor="rgba(0,176,246,0.2)",
                line={"color": "black"},
                name="4.5x LTI",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=budget_l,
                fill="tonexty",
                fillcolor="rgba(0,176,246,0.2)",
                line={"color": "black", "width": 0.8},
                name="4.3x LTI",
            )
        )

        # Work out the values on the target date
        end_date = datetime.datetime.strptime(re.split("T| ", target_date)[0], "%Y-%m-%d")
        start_date = datetime.datetime.today()
        num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        mortgage_l, mortgage_h = 4.3 * income * 1000, 4.7 * income * 1000
        savings_on_target_date = npf.fv((r / 100) / 12, num_months, -saving_rate, -(savings * 1000))

        # Work out deposit, stamp duty and max affordable price for each mortgage size
        deposit_l, property_value_l, stamp_duty_l = iterative_p(savings_on_target_date, mortgage_l, stamp_duty_payable,
                                                                higher_rate)
        deposit_h, property_value_h, stamp_duty_h = iterative_p(savings_on_target_date, mortgage_h, stamp_duty_payable,
                                                                higher_rate)

        deposit_on_target_date_str = f"£{int(deposit_l): ,} - £{int(deposit_h): ,}"
        mortgage_on_target_date_str = f"£{int(mortgage_l): ,} - £{int(mortgage_h): ,}"
        property_value_on_target_date_str = f"£{int(property_value_l): ,} - £{int(property_value_h): ,}"
        stamp_duty_on_target_date_str = f"£{int(stamp_duty_l): ,} - £{int(stamp_duty_h): ,}"

        data_storage = {
            "deposit": [deposit_l, deposit_h],
            "mortgage": [mortgage_l, mortgage_h],
            "value": [property_value_l, property_value_h],
            "stamp_duty": [stamp_duty_l, stamp_duty_h],
            "income": income,
        }
        data_storage = json.dumps(data_storage)

        return (
            fig,
            deposit_on_target_date_str,
            mortgage_on_target_date_str,
            property_value_on_target_date_str,
            stamp_duty_on_target_date_str,
            data_storage,
        )
    else:
        raise PreventUpdate


def stamp_duty_payable(price: int, higher_rate: bool) -> float:
    """
    Computes the stamp duty payable where property costs price.

    Args:
        price (int): price of property
        higher_rate (bool): True if higher rate of stamp duty is payable

    Returns:
        (int) total stamp duty payable
    """
    lease_premiums = [125000, 125000, 675000, 575000, np.inf]
    rates = [0, 0.02, 0.05, 0.10, 0.12]
    if higher_rate:
        rates = [rate + 0.03 for rate in rates]
    payable = 0
    for i, (premium, rate) in enumerate(zip(lease_premiums, rates)):
        # is price above sum of premiums up to and including premium?
        if price > sum(lease_premiums[: i + 1]):
            # if so add premium * rate to payable
            payable += premium * rate
        else:
            payable += max(0, (price - sum(lease_premiums[:i])) * rate)
    return payable


def iterative_p(s, m, stamp_duty_payable, higher_rate):
    """
    Iteratively works out the maximum price affordable inclusive of stamp duty.

    Args:
        s: total savings
        m: mortage size available
        stamp_duty_payable: function to compute stamp_duty
        higher_rate: bool to indicate which stamp duty is payable

    Returns:
        deposit size, maximum price, stamp duty payable
    """
    old_sd, sd = 0, 100
    while abs(old_sd - sd) > 0.01:
        p = s - old_sd + m
        sd = stamp_duty_payable(p, higher_rate)
        new_p = s - sd + m
        old_sd = sd
        sd = stamp_duty_payable(new_p, higher_rate)

    deposit = s - sd
    return deposit, p, sd
