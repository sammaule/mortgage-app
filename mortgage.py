"""Contains layout and callbacks for the mortgage page of the app."""
import json
from typing import Tuple, Union, Optional

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import numpy as np
from dash.exceptions import PreventUpdate

from app import app


first_card = dbc.Card(
    [
        dbc.CardHeader("Mortgage details:"),
        dbc.CardBody(
            [
                dbc.FormGroup(
                    [dbc.Label("Deposit size (£ ,000)"), dbc.Input(id="deposit-size", type="number"), ]
                ),
                dbc.FormGroup(
                    [dbc.Label("Purchase price (£ ,000)"), dbc.Input(id="purchase-price", type="number"), ]
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
                            id="initial-interest-rate", value=1.05, type="number", min=0, max=100, step=0.01,
                        ),
                    ]
                ),
                dbc.FormGroup(
                    [
                        dbc.Label("Interest rate (%): "),
                        dbc.Input(id="interest-rate", value=3.00, type="number", min=0, max=100, step=0.01,),
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
                html.Div([dbc.Label("Total interest payable:"), html.H5(id="total-repaid")]),
                html.Div([dbc.Label("LTV:"), html.H5(id="ltv")]),
                html.Div([dbc.Label("LTI:"), html.H5(id="lti-mortgage")]),
                html.Div(
                    [dbc.Label("Monthly payment (offer period):"), html.H5(id="monthly-payment-offer"), ]
                ),
                html.Div([dbc.Label("Monthly payment:"), html.H5(id="monthly-payment")]),
                html.Br(),
                html.Div(
                    [
                        dbc.Button("Save mortgage", color="primary", size="lg", id="save-button-mortgage"),
                        dbc.Modal(
                            [
                                dbc.ModalHeader("Mortgage saved"),
                                dbc.ModalBody("Go to asset allocation page to view."),
                            ],
                            id="mortgage-saved-popup",
                        ),
                    ]
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
    [Output("deposit-size", "value"), Output("purchase-price", "value"), ],
    [Input("url", "pathname")],
    [State("data-store", "data")],
)
def fill_data_values(url, data) -> Tuple[float, float]:
    """
    Fills deposit and property value data input in budget page.

    Args:
        url: callback triggered by change in url
        data: json string

    Returns:
        deposit size
        purchase price
    """
    if data:
        data = json.loads(data)
        deposit = round(data.get("deposit") / 1000, 3)
        value = round(data.get("value") / 1000, 3)
        return deposit, value


@app.callback(
    [Output("ltv", "children"), Output("mortgage-size", "children"), Output("lti-mortgage", "children")],
    [Input("deposit-size", "value"), Input("purchase-price", "value")],
    [State("data-store", "data")],
)
def calc_mortgage_data(deposit: int, price: int, data: str) -> Tuple[str, str, str]:
    """
    Calculates LTI, LTV and mortgage size based on input price, deposit and income data.

    Args:
        deposit:  deposit size (£k)
        price: price (£k)
        data: JSON str of stored data from budget page

    Returns:
        LTV str
        LTI str
        mortgage size str
    """
    if all(v is not None for v in [deposit, price]):
        data = json.loads(data)
        ltv = round((price - deposit) * 100 / price, 1)
        lti = (price - deposit) / data.get("income")
        ltv_str = f"{ltv}%"
        lti_str = f"{round(lti, 1)}"
        mortgage_str = f"£{int(1000 * (price - deposit)) :,}"
        return ltv_str, mortgage_str, lti_str
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
    deposit: int, purchase_price: int, term: int, interest_rate: float, offer_term: int, offer_rate: float,
) -> Tuple[dict, str, str, str]:
    """
    Callback to plot the payment schedule and populate the total interest repaid.

    Args:
        deposit: deposit amount (£k)
        purchase_price: price (£k)
        term: mortgage length (years)
        interest_rate: interest rate (% annual)
        offer_term: length of introductory offer (years)
        offer_rate: interest rate during offer period (% annual)

    Returns:

    """
    if all(v is not None for v in [deposit, purchase_price, interest_rate, offer_term, offer_rate]):
        total_borrowed = (purchase_price - deposit) * 1000

        # Compute initial monthly payment
        offer_m_payment = calc_monthly_payment(total_borrowed, offer_rate, term)
        initial_payments = np.array([offer_m_payment] * offer_term * 12)

        # remaining balance on loan
        balance = compute_remaining_balance(total_borrowed, offer_rate, offer_term, term)

        # Compute monthly payment according to formula here:
        remaining_term = term - offer_term
        m_payment = calc_monthly_payment(balance, interest_rate, remaining_term)
        later_payments = np.array([m_payment] * remaining_term * 12)

        # Generate plot data
        x = np.array(range(1, term * 12 + 1))
        y = np.append(initial_payments, later_payments)

        # Create figure dict
        figure = {
            "data": [{"x": x, "y": y, "type": "bar", }, ],
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


@app.callback(
    [Output("data-store-mortgage", "data"), Output("mortgage-saved-popup", "is_open")],
    [Input("save-button-mortgage", "n_clicks"), ],
    [
        State("deposit-size", "value"),
        State("purchase-price", "value"),
        State("mortgage-term", "value"),
        State("interest-rate", "value"),
        State("offer-term", "value"),
        State("initial-interest-rate", "value"),
        State("data-store-mortgage", "data"),
    ],
)
def save_mortgage_info(
    n_clicks: Optional[int],
    deposit: int,
    purchase_price: int,
    term: int,
    interest_rate: float,
    offer_term: int,
    offer_rate: float,
    data: Optional[str],
) -> Tuple[str, bool]:
    """
    Callback that stores the mortgage data displayed on page to the data-store-mortgage session memory on
    click of save-button-mortgage. Opens modal window to confirm data saved.

    Args:
        n_clicks: number of times button clicked. None if not clicked.
        deposit: mortgage deposit size (£ thousands)
        purchase_price: house price (£ thousands)
        term: mortgage length (years)
        interest_rate: mortgage interest rate after offer period
        offer_term: mortgage offer term (years)
        offer_rate: mortgage offer rate
        data: JSON str of data-store-mortgage session memory

    Returns:
        JSON str of data-store-mortgage session memory
        bool to determine if mortgage-saved-popup window is open
    """
    if n_clicks is None:
        raise PreventUpdate
    else:
        mortgage_data = {
            "deposit": deposit,
            "mortgage_size": purchase_price - deposit,
            "term": term,
            "interest_rate": interest_rate,
            "offer_term": offer_term,
            "offer_rate": offer_rate,
        }
        if data is None:
            data = json.dumps([mortgage_data])
            return data, True
        else:
            existing_data = json.loads(data)
            # If the mortgage data hasn't been added before append it
            if all(i != mortgage_data for i in existing_data):
                existing_data.append(mortgage_data)

            existing_data = json.dumps(existing_data)
            return existing_data, True


def calc_monthly_payment(total_borrowed: Union[int, float], r: float, term: int) -> float:
    """
    Function to compute monthly mortgage repayments.
    https://en.wikipedia.org/wiki/Mortgage_calculator

    Args:
        total_borrowed: total amount borrowed
        r: annual interest rate (%)
        term: length of mortgage (years)
    Returns:
        monthly payment
    """
    # Convert from years to months
    n_payments = term * 12
    # Convert rate to decimal monthly
    offer_r = (r / 12) / 100
    return (total_borrowed * offer_r) / (1 - (1 + offer_r) ** -n_payments)


def compute_remaining_balance(total_borrowed: int, r: float, offer_term: int, term: int) -> float:
    """
    Computes remaining balance left on loan after end of offer_term periods

    Args:
        total_borrowed: total amount initially borrowed
        r: interest rate paid (%)
        offer_term: length of offer period
        term: total length of loan

    Returns:
        remaining balance on loan
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
