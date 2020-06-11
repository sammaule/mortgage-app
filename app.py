""" entry point for app."""
from typing import Tuple, Dict, Union

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from dash.exceptions import PreventUpdate

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}

app.layout = html.Div(
    [
        dcc.Graph(id="mortgage-plot",),
        # TODO: Align the input boxes
        html.Div(
            [
                "Deposit size (£k)",
                dcc.Input(id="deposit-size", value=150, type="number"),
            ]
        ),
        html.Div(
            [
                "Purchase price (£k)",
                dcc.Input(id="purchase-price", value=375, type="number"),
            ]
        ),
        html.Div(id="ltv"),
        html.Div(
            [
                "Offer term (years): ",
                dcc.Input(id="offer-term", value=2, type="number", min=0),
            ]
        ),
        html.Div(
            [
                "Mortgage term (years): ",
                dcc.Input(id="mortgage-term", value=20, type="number", min=1),
            ]
        ),
        html.Div(
            [
                "Initial interest rate (%): ",
                dcc.Input(
                    id="initial-interest-rate",
                    value=1.05,
                    type="number",
                    min=0.01,
                    max=100,
                ),
            ]
        ),
        html.Div(
            [
                "Interest rate (%): ",
                dcc.Input(
                    id="interest-rate", value=3.0, type="number", min=0.01, max=100
                ),
            ]
        ),
        html.Div(id="total-repaid"),
    ]
)


@app.callback(
    Output("ltv", "children"),
    [Input("deposit-size", "value"), Input("purchase-price", "value")],
)
def calc_ltv(deposit: int, price: int) -> str:
    """
    Returns LTV of mortgage.

    Args:
        deposit (): int deposit size (£k)
        price (): int price (£k)

    Returns:
        str LTV for display in div
    """
    ltv = round((price - deposit) * 100 / price, 1)
    return f"LTV: {ltv}%"


@app.callback(
    [Output("mortgage-plot", "figure"), Output("total-repaid", "children")],
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
) -> Tuple[dict, str]:
    """
    Callback to plot the payment schedule and populate the total interest repaid.

    Args:
        deposit (): int deposit amount (£k)
        purchase_price (): int price (£k)
        term (): int mortgage length (years)
        interest_rate (): float interest rate (% annual)
        offer_term (): int length of introductory offer (years)
        offer_rate (): float

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
        m_payments = calc_monthly_payment(balance, interest_rate, remaining_term)
        later_payments = np.array([m_payments] * remaining_term * 12)

        # Generate plot data
        x = np.array(range(1, term * 12 + 1))
        y = np.append(initial_payments, later_payments)

        # Create figure dict
        figure = {
            "data": [{"x": x, "y": y, "type": "bar",},],
            "layout": {
                "title": "Mortgage repayment schedule",
                "xaxis": {"title": "Months"},
                "yaxis": {"title": "Monthly payment (£)"},
                "clickmode": "event+select",
            },
        }
        interest_paid = int(np.sum(y) - total_borrowed)
        interest_paid = f"Total interest paid: £{interest_paid:,}"
        return figure, interest_paid
    else:
        raise PreventUpdate


def calc_monthly_payment(total_borrowed: Union[int, float], r: float, term: int) -> float:
    """
    Function to compute monthly mortgage repayments.
    https://en.wikipedia.org/wiki/Mortgage_calculator

    Args:
        total_borrowed (): total amount borrowed
        r (): annual interest rate (%)
        term (): length of mortgage (years)
    Returns:
        float monthly payment
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
        total_borrowed (): total amount initially borrowed
        r (): interest rate paid (%)
        offer_term (): length of offer period
        term (): total length of loan

    Returns:
        float remaining balance on loan
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


@app.callback(Output("offer-term", "max"), [Input("mortgage-term", "value")])
def limit_offer_term(mortgage_term: int) -> int:
    """
    Callback to ensure offer term is shorter than mortgage term

    Args:
        mortgage_term (): int mortgage term (years)

    Returns:
        Output('offer-term', 'max') : int max value for offer term
    """
    if mortgage_term is not None:
        return mortgage_term - 1
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run_server(debug=True)
