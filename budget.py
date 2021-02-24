"""Code for the budget page of the app."""
import datetime
import json

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
                        dbc.Label("Total savings (£ ,000)"),
                        dbc.Input(id="current-savings", value=183, type="number"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Monthly savings rate (£)"),
                        dbc.Input(id="saving-rate", value=800, step=10, type="number",),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Savings interest (% annual)"),
                        dbc.Input(id="savings-interest", value=0.5, step=0.01, type="number",),
                    ]
                ),
            ]
        ),
    ]
)

income_and_tax_card = dbc.Card(
    [
        dbc.CardHeader("Other info"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [dbc.Label("Income (£ ,000)"), dbc.Input(id="income", value=60, type="number"), ]
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
                        dbc.Label("Loan to income ratio"),
                        dcc.Slider(
                            id="lti",
                            value=4,
                            marks={3: "3", 3.5: "3.5", 4: "4", 4.5: "4.5", 5: "5"},
                            step=0.1,
                            min=3,
                            max=5,
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
                            date=datetime.datetime.today().date() + datetime.timedelta(days=9 * 30),
                            min_date_allowed=datetime.datetime.today(),
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
        dbc.CardHeader("Maximum affordability breakdown"),
        dbc.CardBody(
            [
                html.Div(
                    [dbc.Label("Maximum affordable value:"), html.H5(id="property-value-on-target-date"), ]
                ),
                html.Div([dbc.Label("Mortgage size:"), html.H5(id="mortgage-on-target-date")]),
                html.Div([dbc.Label("Deposit size:"), html.H5(id="deposit-on-target-date")]),
                html.Div([dbc.Label("Stamp duty payable:"), html.H5(id="stamp-duty-on-target-date")]),
            ]
        ),
    ]
)

layout = html.Div(
    [
        dbc.Row([dbc.Col([savings_card], width=6), dbc.Col([income_and_tax_card], width=6)]),
        dbc.Row(
            [
                dbc.Col([budget_results_card], width=5),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Affordability to target date"),
                                dbc.CardBody(dcc.Graph(id="budget-plot")),
                            ]
                        )
                    ],
                    width=7,
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
        Input("lti", "value"),
        Input("target-purchase-date", "date"),
    ],
)
def calc_ltv(
    savings: int, saving_rate: int, r: float, income: int, stamp_duty_rate: str, lti: float, target_date: str
) -> Tuple[go.Figure, str, str, str, str, str]:
    """
    Callback to populate data in the savings plot according to the input values entered by user.
    Assumes that all interest is paid monthly at a rate of 1/12 * r and all reinvested.

    Args:
        savings: Total current savings
        saving_rate: Amount saved each month
        r: interest rate (%)
        income: user input income (£ thousands)
        stamp_duty_rate: one of ["higher_rate", "lower_rate"]
        lti: loan to income ratio
        target_date: purchase date with format YYYY-MM-DD
    """
    if all(v is not None for v in [savings, saving_rate, r]):
        fig = go.Figure()

        start_date = datetime.datetime.today().date()
        end_date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        x = pd.date_range(datetime.datetime.now(), periods=periods, freq="M")

        savings_array = np.array(
            [npf.fv((r / 100) / 12, i, -saving_rate, -(savings * 1000)) for i in range(periods)]
        )

        mortgage = lti * (income * 1000)

        higher_rate = True if stamp_duty_rate == "higher_rate" else False

        budget = np.array(
            [iterative_p(s, mortgage, stamp_duty_payable, higher_rate)[1] for s in savings_array]
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=budget,
                fillcolor="rgba(0,176,246,0.2)",
                line={"color": "black", "width": 0.8},
                hovertemplate="Date: %{x} \nMax affordable: £%{y:,.3r}<extra></extra>",
            )
        )

        # Work out the values on the target date

        start_date = datetime.datetime.today()
        num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

        mortgage = lti * income * 1000
        savings_on_target_date = npf.fv((r / 100) / 12, num_months, -saving_rate, -(savings * 1000))

        # Work out deposit, stamp duty and max affordable price for each mortgage size
        deposit, property_value, stamp_duty = iterative_p(
            savings_on_target_date, mortgage, stamp_duty_payable, higher_rate
        )

        deposit_on_target_date_str = f"£{int(deposit): ,}"
        mortgage_on_target_date_str = f"£{int(mortgage): ,}"
        property_value_on_target_date_str = f"£{int(property_value): ,}"
        stamp_duty_on_target_date_str = f"£{int(stamp_duty): ,}"

        data_storage = {
            "savings": savings_on_target_date / 1000,
            "deposit": deposit,
            "mortgage": mortgage,
            "value": property_value,
            "stamp_duty": stamp_duty,
            "stamp_duty_rate": stamp_duty_rate,
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
        price: price of property
        higher_rate: True if higher rate of stamp duty is payable

    Returns:
        total stamp duty payable
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


def iterative_p(s, m, stamp_duty_payable_fn, higher_rate):
    """
    Iteratively works out the maximum price affordable inclusive of stamp duty.

    Args:
        s: total savings
        m: mortgage size available
        stamp_duty_payable_fn: function to compute stamp_duty
        higher_rate: bool to indicate which stamp duty is payable

    Returns:
        deposit size, maximum price, stamp duty payable
    """
    old_sd, sd, p = 0, 100, 0
    while abs(old_sd - sd) > 0.01:
        p = s - old_sd + m
        sd = stamp_duty_payable_fn(p, higher_rate)
        new_p = s - sd + m
        old_sd = sd
        sd = stamp_duty_payable_fn(new_p, higher_rate)

    deposit = s - sd
    return deposit, p, sd
