"""Code for the budget page of the app."""
import datetime

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
                        dbc.Input(
                            id="savings-interest", value=1.12, step=0.01, type="number",
                        ),
                    ]
                ),
            ]
        ),
    ]
)

income_and_tax_card = dbc.Card(
    [
        dbc.CardHeader("Income and Stamp Duty"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
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
            ]
        ),
    ]
)

layout = html.Div(
    [
        dbc.Row(
            [dbc.Col([savings_card], width=6), dbc.Col([income_and_tax_card], width=6)]
        ),
        dbc.Row(
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Maximum budget"),
                            dbc.CardBody(dcc.Graph(id="budget-plot")),
                        ]
                    )
                ],
                width=12,
            ),
        ),
    ]
)

@app.callback(
    Output("budget-plot", "figure"),
    [
        Input("current-savings", "value"),
        Input("saving-rate", "value"),
        Input("savings-interest", "value"),
        Input("income", "value"),
        Input("stamp-duty-rate", "value"),
    ],
)
def calc_ltv(
    savings: int, saving_rate: int, r: float, income: int, stamp_duty_rate: str
) -> dict:
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

        savings = np.array(
            [
                npf.fv((r / 100) / 12, i, -saving_rate, -(savings * 1000))
                for i in range(24)
            ]
        )

        budget_h = savings + 4.7 * (income * 1000)
        budget_m = savings + 4.5 * (income * 1000)
        budget_l = savings + 4.3 * (income * 1000)

        higher_rate = True if stamp_duty_rate == "higher_rate" else False
        budget_h = np.array(
            [
                price - stamp_duty_payable(price, higher_rate=higher_rate)
                for price in budget_h
            ]
        )
        budget_m = np.array(
            [
                price - stamp_duty_payable(price, higher_rate=higher_rate)
                for price in budget_m
            ]
        )
        budget_l = np.array(
            [
                price - stamp_duty_payable(price, higher_rate=higher_rate)
                for price in budget_l
            ]
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
        return fig
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