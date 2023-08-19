"""Contains layout and callbacks for the mortgage page of the app."""
import logging
import json
from typing import Optional, Tuple

from dash import callback, dash_table
import dash_bootstrap_components as dbc
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# TODO: Get logs working
logger = logging.getLogger(__name__)

mortgage_input_card = dbc.Card(
    [
        dbc.CardHeader("Mortgage details:"),
        dbc.CardBody(
            [
                dbc.Label("Deposit size (£k)"),
                dbc.Input(id="deposit-size", type="number"),
                dbc.Label("Purchase price (£k)"),
                dbc.Input(id="purchase-price", type="number"),
                dbc.Label("Offer term (years): "),
                dcc.Slider(
                    id="offer-term",
                    value=5,
                    marks={i: f"{i}" for i in range(0, 11)},
                    step=1,
                    min=0,
                    max=10,
                ),
                dbc.Label("Mortgage term (years): "),
                dcc.Slider(
                    id="mortgage-term",
                    value=25,
                    marks={i: f"{i}" for i in range(5, 41, 5)},
                    step=1,
                    min=5,
                    max=40,
                ),
                dbc.Label("Estimated years in property:"),
                dcc.Slider(
                    id="years-in-property",
                    value=10,
                    marks={i: f"{i}" for i in range(5, 41, 5)},
                    step=1,
                    min=1,
                    max=40,
                ),
                dbc.Label("Initial interest rate (%): "),
                dbc.Input(
                    id="initial-interest-rate",
                    value=5.75,
                    type="number",
                    min=0,
                    max=100,
                    step=0.01,
                ),
                dbc.Label("Interest rate (%): "),
                dbc.Input(
                    id="interest-rate",
                    value=5.75,
                    type="number",
                    min=0,
                    max=100,
                    step=0.01,
                ),
            ]
        ),
    ],
)

mortgage_info_card = dbc.Card(
    [
        dbc.CardHeader("Mortgage info:"),
        dbc.CardBody(
            [
                html.Div([dbc.Label("Mortgage size:"), html.H5(id="mortgage-size")]),
                html.Div([dbc.Label("Total interest payable:"), html.H5(id="total-repaid")]),
                html.Div([dbc.Label("LTV:"), html.H5(id="ltv")]),
                html.Div([dbc.Label("LTI:"), html.H5(id="lti-mortgage")]),
                html.Div(
                    [
                        dbc.Label("Monthly payment (offer period):"),
                        html.H5(id="monthly-payment-offer"),
                    ]
                ),
                html.Div([dbc.Label("Monthly payment:"), html.H5(id="monthly-payment")]),
                html.Br(),
                html.Div(
                    [
                        dbc.Button(
                            "Save mortgage",
                            color="primary",
                            size="lg",
                            id="save-button-mortgage",
                        ),
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

cost_table_div = html.Div(id="cost-table-div")

layout = html.Div(
    [
        dcc.Store(id="mortgage-payments-store", storage_type="session"),
        dbc.Row([dbc.Col(mortgage_input_card, width=6), dbc.Col(mortgage_info_card, width=6)]),
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
        dbc.Row(cost_table_div),
    ],
)


@callback(
    [
        Output("deposit-size", "value"),
        Output("purchase-price", "value"),
    ],
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
        deposit = round(data.get("deposit") / 1000, 1)
        value = round(data.get("value") / 1000, 1)
        return deposit, value


@callback(
    [
        Output("ltv", "children"),
        Output("mortgage-size", "children"),
        Output("lti-mortgage", "children"),
    ],
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


@callback(
    Output("mortgage-payments-store", "data"),
    [
        Input("deposit-size", "value"),
        Input("purchase-price", "value"),
        Input("mortgage-term", "value"),
        Input("interest-rate", "value"),
        Input("offer-term", "value"),
        Input("initial-interest-rate", "value"),
    ],
)
def store_mortgage_payments(
    deposit: int,
    purchase_price: int,
    term: int,
    interest_rate: float,
    offer_term: int,
    offer_rate: float,
) -> str:
    """
    Callback to store mortgage payments data in session memory.

    Args:
        deposit: deposit amount (£k)
        purchase_price: price (£k)
        term: mortgage length (years)
        interest_rate: interest rate (% annual)
        offer_term: length of introductory offer (years)
        offer_rate: interest rate during offer period (% annual)

    Returns:
        JSON str of mortgage payments data
    """
    if all(v is not None for v in [deposit, purchase_price, interest_rate, offer_term, offer_rate]):
        purchase_price = purchase_price * 1000
        deposit = deposit * 1000
        total_borrowed = purchase_price - deposit
        offer_rate = (offer_rate / 12) / 100  # monthly interest rate
        interest_rate = (interest_rate / 12) / 100
        # Convert terms from years to months
        term *= 12
        offer_term *= 12
        remaining_term = term - offer_term
        month = np.arange(term) + 1

        # Compute the mortgage payments, interest payments and principal payments during the offer period
        # These functions return negative values so multiply by -1
        offer_payments = -1 * npf.pmt(offer_rate, term, total_borrowed)
        offer_interest_payments = (-1 * npf.ipmt(offer_rate, month, term, total_borrowed))[:offer_term]
        offer_principal_payments = (-1 * npf.ppmt(offer_rate, month, term, total_borrowed))[:offer_term]

        # Same for the remaining term
        balance_after_offer = total_borrowed - np.sum(offer_principal_payments)
        per = np.arange(remaining_term) + 1
        remaining_payments = -1 * npf.pmt(interest_rate, remaining_term, balance_after_offer)
        remaining_interest_payments = -1 * npf.ipmt(interest_rate, per, remaining_term, balance_after_offer)
        remaining_principal_payments = -1 * npf.ppmt(interest_rate, per, remaining_term, balance_after_offer)

        # Combine the offer and remaining term payments
        interest_payments = np.append(offer_interest_payments, remaining_interest_payments)
        principal_payments = np.append(offer_principal_payments, remaining_principal_payments)

        assert np.isclose(np.sum(principal_payments), total_borrowed)

        mortgage_payments_data = json.dumps(
            {
                "purchase_price": purchase_price,
                "deposit": deposit,
                "total_borrowed": total_borrowed,
                "month": month.tolist(),
                "offer_payments": offer_payments.tolist(),
                "regular_payments": remaining_payments.tolist(),
                "interest_payments": interest_payments.tolist(),
                "principal_payments": principal_payments.tolist(),
            }
        )
        return mortgage_payments_data
    else:
        raise PreventUpdate


@callback(
    [
        Output("monthly-payment-offer", "children"),
        Output("monthly-payment", "children"),
        Output("total-repaid", "children"),
    ],
    Input("mortgage-payments-store", "data"),
)
def calc_repayment_strs(mortgage_payments_data: str) -> Tuple[str, str, str]:
    """Fills the total interest payable, monthly payment and monthly payment during offer period

    Args:
        mortgage_payments_data (str): stored mortgage payments data

    Raises:
        PreventUpdate: _description_

    Returns:
        Tuple[str, str, str]: _description_
    """
    mortgage_data = json.loads(mortgage_payments_data)
    offer_payment = mortgage_data.get("offer_payments", 0)
    regular_payment = mortgage_data.get("regular_payments", 0)
    total_payment = sum(mortgage_data.get("interest_payments"))
    return f"£{int(offer_payment) :,}", f"£{int(regular_payment) :,}", f"£{int(total_payment) :,}"


@callback(
    Output("mortgage-plot", "figure"),
    Input("mortgage-payments-store", "data"),
)
def plot_monthly_repayments(
    mortgage_payments_data: str,
) -> go.Figure:
    """
    Callback to plot the monthly mortgage payments.

    Args:
        mortgage_payments_data: JSON str of mortgage payments data

    Returns:
        Plotly figure
    """
    # read json data
    mortgage_payments_data = json.loads(mortgage_payments_data)

    # Create figure dict
    figure = go.Figure(
        data=[
            go.Bar(
                x=mortgage_payments_data.get("month"),
                y=mortgage_payments_data.get("interest_payments"),
                name="Interest payments",
            ),
            go.Bar(
                x=mortgage_payments_data.get("month"),
                y=mortgage_payments_data.get("principal_payments"),
                name="Principal payments",
            ),
        ],
    )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(title_text="Months")
    figure.update_yaxes(title_text="Total payment (£)")

    return figure


@callback(Output("cost-table-div", "children"), [Input("mortgage-payments-store", "data"), Input("years-in-property", "value")])
def create_cost_table(mortgage_payments_data: str, years_in_property: int) -> dash_table.DataTable:
    """
    Callback to create the mortgage cost table.

    Args:
        mortgage_payments_data: JSON str of mortgage payments data

    Returns:
        List of dicts to populate the table
    """
    logger.debug("Creating mortgage cost table")
    mortgage_payments_data = json.loads(mortgage_payments_data)
    month = mortgage_payments_data.get("month")
    interest_payments = mortgage_payments_data.get("interest_payments")
    principal_payments = mortgage_payments_data.get("principal_payments")
    total_payments = np.array(interest_payments) + np.array(principal_payments)

    # TODO: Add amortised stamp duty payments, maintenance costs, etc.
    yearly_mainenance_costs = mortgage_payments_data.get("purchase_price") * 0.01
    maintenance_costs = np.empty(years_in_property * 12)
    maintenance_costs.fill(yearly_mainenance_costs / 12)
    if len(month) - len(maintenance_costs) > 0:
        maintenance_costs = np.append(maintenance_costs, np.zeros(len(month) - len(maintenance_costs)))

    return dash_table.DataTable(
        data=[
            {
                "Month": round(month[i], 2),
                "Interest payments": round(interest_payments[i], 2),
                "Principal payments": round(principal_payments[i], 2),
                "Total payments": round(total_payments[i], 2),
                "Maintenance costs": round(maintenance_costs[i], 2),
            }
            for i in range(len(month))
        ],
        columns=[
            {"name": "Month", "id": "Month"},
            {"name": "Interest payments", "id": "Interest payments"},
            {"name": "Principal payments", "id": "Principal payments"},
            {"name": "Total payments", "id": "Total payments"},
            {"name": "Maintenance costs", "id": "Maintenance costs"},
        ],
        export_format="csv",
    )


@callback(
    [Output("data-store-mortgage", "data"), Output("mortgage-saved-popup", "is_open")],
    [
        Input("save-button-mortgage", "n_clicks"),
    ],
    [
        State("deposit-size", "value"),
        State("purchase-price", "value"),
        State("mortgage-term", "value"),
        State("interest-rate", "value"),
        State("offer-term", "value"),
        State("initial-interest-rate", "value"),
        State("mortgage-payments-store", "data"),
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
    mortgage_payments_data: str,
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
        mortgage_payments_data = json.loads(mortgage_payments_data)
        offer_payment = mortgage_payments_data.get("offer_payments", 0)
        regular_payment = mortgage_payments_data.get("regular_payments", 0)

        mortgage_data = {
            "deposit": deposit,
            "mortgage_size": purchase_price - deposit,
            "purchase_price": purchase_price,
            "term": term,
            "interest_rate": interest_rate,
            "offer_term": offer_term,
            "offer_rate": offer_rate,
            "offer_payment": offer_payment,
            "regular_payment": regular_payment,
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
