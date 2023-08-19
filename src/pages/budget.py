"""Code for the budget page of the app."""
import datetime
import json
from bisect import bisect
from typing import Callable, Tuple, Union

from dash import callback
import dash_bootstrap_components as dbc
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from config import stamp_duty_rates

inputs_card = dbc.Card(
    [
        dbc.CardHeader("Savings"),
        dbc.CardBody(
            [
                html.P(
                    "Enter details of your current financial situtation to discover the maximum affordable house price for you at your target purchase date."
                ),
                dbc.Label("Total savings (£k)"),
                dbc.Input(id="current-savings", value=50, type="number"),
                dbc.Label("Monthly savings rate (£)"),
                dbc.Input(
                    id="saving-rate",
                    value=500,
                    step=10,
                    type="number",
                ),
                dbc.Label("Savings interest (% annual)"),
                dbc.Input(
                    id="savings-interest",
                    value=0.5,
                    step=0.01,
                    type="number",
                ),
                dbc.Label("Income (£ ,000)"),
                dbc.Input(id="income", value=25, type="number"),
                dbc.Label("Stamp Duty rate"),
                dbc.Select(
                    id="stamp-duty-rate",
                    options=[
                        {"label": "Normal rate", "value": "normal_rate"},
                        {"label": "Higher rate", "value": "higher_rate"},
                        {
                            "label": "First time buyer rate",
                            "value": "first_time_rate",
                        },
                    ],
                    value="normal_rate",
                ),
                dbc.Label("Mortgage loan to income ratio"),
                dcc.Slider(
                    id="lti",
                    value=4,
                    marks={3: "3", 3.5: "3.5", 4: "4", 4.5: "4.5", 5: "5"},
                    step=0.1,
                    min=3,
                    max=5,
                ),
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
    ],
)

budget_results_card = dbc.Card(
    [
        dbc.CardHeader("Maximum affordability breakdown"),
        dbc.CardBody(
            [
                html.Div(
                    [
                        dbc.Label("Maximum affordable house price:"),
                        html.H5(id="property-value-on-target-date"),
                    ]
                ),
                html.Div([dbc.Label("Mortgage size:"), html.H5(id="mortgage-on-target-date")]),
                html.Div([dbc.Label("Deposit size:"), html.H5(id="deposit-on-target-date")]),
                html.Div(
                    [
                        dbc.Label("Stamp duty payable:"),
                        html.H5(id="stamp-duty-on-target-date"),
                    ]
                ),
            ]
        ),
    ]
)

layout = html.Div(
    [
        dbc.Row([dbc.Col([inputs_card], width=6), dbc.Col([budget_results_card], width=6)]),
        dbc.Row(
            [
                dbc.Card(
                    [
                        dbc.CardHeader("Affordability to target date"),
                        dbc.CardBody(dcc.Graph(id="budget-plot")),
                    ]
                )
            ],
        ),
    ]
)


@callback(
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
def plot_affordability(
    savings: int,
    saving_rate: int,
    r: float,
    income: int,
    rate_type: str,
    lti: float,
    target_date: str,
) -> Tuple[go.Figure, str, str, str, str, str]:
    """
    Callback to populate data in the savings plot according to the input values entered by user.
    Assumes that all interest is paid monthly at a rate of 1/12 * r and all reinvested.

    Stores data to data-store.

    Args:
        savings: Total current savings
        saving_rate: Amount saved each month
        r: interest rate (%)
        income: user input income (£ thousands)
        rate_type: type of stamp duty rate payable
        lti: loan to income ratio
        target_date: purchase date with format YYYY-MM-DD
    """
    if all(v is not None for v in [savings, saving_rate, r, income, lti, target_date]):
        fig = go.Figure()

        start_date = datetime.datetime.today().date()
        target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        periods = (target_date.year - start_date.year) * 12 + (target_date.month - start_date.month) + 1
        x = pd.date_range(datetime.datetime.now(), periods=periods, freq="M")

        savings_array = np.array([npf.fv((r / 100) / 12, i, -saving_rate, -(savings * 1000)) for i in range(periods)])

        mortgage = lti * (income * 1000)

        budget = np.array([iterative_p(s, mortgage, stamp_duty_payable, rate_type, target_date)[1] for s in savings_array])

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
        num_months = (target_date.year - start_date.year) * 12 + (target_date.month - start_date.month)

        mortgage = lti * income * 1000
        savings_on_target_date = npf.fv((r / 100) / 12, num_months, -saving_rate, -(savings * 1000))

        # Work out deposit, stamp duty and max affordable price for each mortgage size
        deposit, property_value, stamp_duty = iterative_p(savings_on_target_date, mortgage, stamp_duty_payable, rate_type, target_date)

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
            "stamp_duty_rate": rate_type,
            "income": income,
            "target_date": target_date.strftime("%Y-%m-%d"),
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


def stamp_duty_payable(target_date: datetime.datetime, price: Union[int, float], rate_type: str) -> float:
    """
    Computes the stamp duty payable where property costs price.

    Args:
        target_date: user's target purchase date
        price: price of property
        rate_type: type of stamp duty rate payable

    Returns:
        total stamp duty payable
    """
    rates = _get_stamp_duty_rates(target_date, stamp_duty_rates)

    thresholds = np.array(rates.get("thresholds"))
    rates = np.array(rates.get(rate_type))

    # Get marginal amounts payable at rate at each threshold
    marginal_amounts = thresholds - np.append(np.zeros(1), thresholds)[:-1]
    # Find threshold above price
    idx = bisect(thresholds, price)
    if idx == 0:
        payable = rates[idx] * price
    else:
        # Calculate payable at brackets below price
        base = np.dot(rates[:idx], marginal_amounts[:idx])
        # and at bracket where price falls
        at_bracket = (price - thresholds[idx - 1]) * rates[idx]
        payable = base + at_bracket
    return payable


def _get_stamp_duty_rates(target_date, sd_rates):
    """Returns dict of rates and thresholds for planned purchase date from config stamp duty rates."""
    for rates in sd_rates:
        start_date, end_date = rates.get("date_range")
        if start_date <= target_date <= end_date:
            return rates
        else:
            continue
    raise ValueError("Stamp duty rates for target date could not be found.")


def iterative_p(
    s: int,
    m: float,
    stamp_duty_payable_fn: Callable[[datetime.datetime, Union[int, float], str], float],
    rate_type: str,
    target_date: datetime.datetime,
) -> Tuple[float, float, float]:
    """
    Iteratively works out the maximum price affordable inclusive of stamp duty.

    Args:
        s: total savings
        m: mortgage size available
        stamp_duty_payable_fn: function to compute stamp_duty
        rate_type: str indicating which stamp duty rate is payable
        target_date: user's target purchase date

    Returns:
        deposit size, maximum price, stamp duty payable
    """
    old_sd, sd, p = 0, 100, 0
    while abs(old_sd - sd) > 0.01:
        p = s - old_sd + m
        sd = stamp_duty_payable_fn(target_date, p, rate_type)
        new_p = s - sd + m
        old_sd = sd
        sd = stamp_duty_payable_fn(target_date, new_p, rate_type)

    deposit = s - sd
    return deposit, p, sd
