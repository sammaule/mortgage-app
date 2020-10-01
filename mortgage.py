""" entry point for app."""
from typing import Tuple, Union

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from dash.exceptions import PreventUpdate

from app import app


first_card = dbc.Card(
    [
        dbc.CardHeader("Mortgage details:"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [
                        dbc.Label("Deposit size (£k)"),
                        dbc.Input(id="deposit-size", value=150, type="number"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Purchase price (£k)"),
                        dbc.Input(id="purchase-price", value=375, type="number"),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Offer term (years): "),
                        dcc.Slider(
                            id="offer-term",
                            value=3,
                            marks={i: f"{i}" for i in range(0, 6)},
                            step=1,
                            min=0,
                            max=5,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Mortgage term (years): "),
                        dcc.Slider(
                            id="mortgage-term",
                            value=20,
                            marks={i: f"{i}" for i in range(5, 41, 5)},
                            step=1,
                            min=5,
                            max=40,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Initial interest rate (%): "),
                        dbc.Input(
                            id="initial-interest-rate",
                            value=1.05,
                            type="number",
                            min=0,
                            max=100,
                            step=0.01,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Interest rate (%): "),
                        dbc.Input(
                            id="interest-rate",
                            value=3.00,
                            type="number",
                            min=0,
                            max=100,
                            step=0.01,
                        ),
                    ]
                ),
            ]
        ),
    ],
)

second_card = dbc.Card(
    [
        dbc.CardHeader("Mortgage info:"),
        dbc.CardBody(
            [
                html.Div([dbc.Label("Mortgage size:"), html.H5(id="mortgage-size")]),
                html.Div(
                    [dbc.Label("Total interest payable:"), html.H5(id="total-repaid")]
                ),
                html.Div([dbc.Label("LTV:"), html.H5(id="ltv")]),
                # TODO: Add an LTI
                html.Div(
                    [
                        dbc.Label("Monthly payment (offer period):"),
                        html.H5(id="monthly-payment-offer"),
                    ]
                ),
                html.Div(
                    [dbc.Label("Monthly payment:"), html.H5(id="monthly-payment")]
                ),
            ]
        ),
    ]
)

layout = html.Div(
    [
        dbc.Row([dbc.Col(first_card, width=6), dbc.Col(second_card, width=6)]),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Mortgage repayment schedule"),
                            dbc.CardBody(dcc.Graph(id="mortgage-plot")),
                        ]
                    ),
                    width=12,
                ),
            ]
        ),
    ],
)


@app.callback(
    [Output("ltv", "children"), Output("mortgage-size", "children")],
    [Input("deposit-size", "value"), Input("purchase-price", "value")],
)
def calc_ltv(deposit: int, price: int) -> Tuple[str, str]:
    """
    Returns LTV of mortgage.

    Args:
        deposit (int):  deposit size (£k)
        price (int): price (£k)

    Returns:
        (str) : LTV for display in div
    """
    if all(
            v is not None
            for v in [deposit, price]
    ):
        ltv = round((price - deposit) * 100 / price, 1)
        return f"{ltv}%", f"£{1000 * (price - deposit) :,}"
    else:
        raise PreventUpdate


@app.callback(
    [
        Output("mortgage-plot", "figure"),
        Output("total-repaid", "children"),
        Output("monthly-payment-offer", "children"),
        Output("monthly-payment", "children"),
    ],
    [
        Input("deposit-size", "value"),
        Input("purchase-price", "value"),
        Input("mortgage-term", "value"),
        Input("interest-rate", "value"),
        Input("offer-term", "value"),
        Input("initial-interest-rate", "value"),
    ],
)
def plot_monthly_repayments(
    deposit: int,
    purchase_price: int,
    term: int,
    interest_rate: float,
    offer_term: int,
    offer_rate: float,
) -> Tuple[dict, str, str, str]:
    """
    Callback to plot the payment schedule and populate the total interest repaid.

    Args:
        deposit (int): deposit amount (£k)
        purchase_price (int): price (£k)
        term (int): mortgage length (years)
        interest_rate (float): interest rate (% annual)
        offer_term (int): length of introductory offer (years)
        offer_rate (float): interest rate during offer period (% annual)

    Returns:

    """
    if all(
        v is not None
        for v in [deposit, purchase_price, interest_rate, offer_term, offer_rate]
    ):
        total_borrowed = (purchase_price - deposit) * 1000

        # Compute initial monthly payment
        offer_m_payment = calc_monthly_payment(total_borrowed, offer_rate, term)
        initial_payments = np.array([offer_m_payment] * offer_term * 12)

        # remaining balance on loan
        balance = compute_remaining_balance(
            total_borrowed, offer_rate, offer_term, term
        )

        # Compute monthly payment according to formula here:
        remaining_term = term - offer_term
        m_payment = calc_monthly_payment(balance, interest_rate, remaining_term)
        later_payments = np.array([m_payment] * remaining_term * 12)

        # Generate plot data
        x = np.array(range(1, term * 12 + 1))
        y = np.append(initial_payments, later_payments)

        # Create figure dict
        figure = {
            "data": [{"x": x, "y": y, "type": "bar",},],
            "layout": {
                "xaxis": {"title": "Months"},
                "yaxis": {"title": "Monthly payment (£)"},
                "clickmode": "event+select",
            },
        }
        # Create strings for output
        interest_paid = int(np.sum(y) - total_borrowed)
        interest_paid = f"£{interest_paid:,}"
        offer_m_payment = f"£{offer_m_payment :,.2f}"
        m_payment = f"£{m_payment :,.2f}"
        return figure, interest_paid, offer_m_payment, m_payment
    else:
        raise PreventUpdate


def calc_monthly_payment(
    total_borrowed: Union[int, float], r: float, term: int
) -> float:
    """
    Function to compute monthly mortgage repayments.
    https://en.wikipedia.org/wiki/Mortgage_calculator

    Args:
        total_borrowed (int): total amount borrowed
        r (float): annual interest rate (%)
        term (int): length of mortgage (years)
    Returns:
        (float) : monthly payment
    """
    # Convert from years to months
    n_payments = term * 12
    # Convert rate to decimal monthly
    offer_r = (r / 12) / 100
    return (total_borrowed * offer_r) / (1 - (1 + offer_r) ** -n_payments)


def compute_remaining_balance(
    total_borrowed: int, r: float, offer_term: int, term: int
) -> float:
    """
    Computes remaining balance left on loan after end of offer_term periods

    Args:
        total_borrowed (int): total amount initially borrowed
        r (float): interest rate paid (%)
        offer_term (int): length of offer period
        term (int): total length of loan

    Returns:
        (float) : remaining balance on loan
    """
    # Formula from https://www.mtgprofessor.com/formulas.htm
    # Convert to decimal monthly interest rate
    offer_r = (r / 12) / 100
    # Convert from years to months
    offer_months = offer_term * 12
    n_payments = term * 12
    # Compute remaining balance
    return (
        total_borrowed
        * ((1 + offer_r) ** n_payments - (1 + offer_r) ** offer_months)
        / ((1 + offer_r) ** n_payments - 1)
    )
