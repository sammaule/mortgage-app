""" entry point for app."""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    dcc.Graph(
        id='mortgage-plot',
    ),
    # TODO: Align the input boxes
    html.Div([
        "Mortgage size (£k)",
        dcc.Input(id='mortgage-size', value=200, type='number'),
    ]),
    html.Div([
        "Purchase price (£k)",
        dcc.Input(id='purchase-price', value=350, type='number'),
    ]),
    html.Div(id="ltv"
             ),
    html.Div([
        "Offer term (years): ",
        # TODO: Write callback so max offer term can't be longer than current mortgage term
        dcc.Input(id='offer-term', value=2, type='number', min=0),
    ]),
    html.Div([
        "Mortgage term (years): ",
        dcc.Input(id='mortgage-term', value=20, type='number', min=0),
    ]),
    html.Div([
        "Initial interest rate (%): ",
        dcc.Input(id='initial-interest-rate', value=1.0, type='number', min=0, max=100),
    ]),
    html.Div([
        "Interest rate (%): ",
        dcc.Input(id='interest-rate', value=2.0, type='number', min=0, max=100),
    ]),
    html.Div(id="total-repaid"
             ),
])


@app.callback(
    Output('ltv', 'children'),
    [Input('mortgage-size', 'value'),
     Input('purchase-price', 'value')]
)
def calc_ltv(mortgage, price):
    ltv = round(mortgage * 100 / price, 1)
    return f"LTV: {ltv}%"


@app.callback(
    [Output('mortgage-plot', 'figure'),
     Output('total-repaid', 'children')
     ],
    [Input('mortgage-size', 'value'),
     Input('mortgage-term', 'value'),
     Input('interest-rate', 'value'),
     Input('offer-term', 'value'),
     Input('initial-interest-rate', 'value'),
     ]
)
def plot_monthly_repayments(total, term, interest_rate, offer_term, offer_rate):
    """
    Callback to plot the payment schedule and populate the total interest repaid.

    Args:
        total ():
        term ():
        interest_rate ():
        offer_term ():
        offer_rate ():

    Returns:

    """
    total_borrowed = total * 1000

    # Compute initial monthly payment
    offer_m_payment = calc_monthly_payment(total_borrowed, offer_rate, term)
    initial_payments = np.array([offer_m_payment] * offer_term * 12)

    # remaining balance on loan
    balance = compute_remaining_balance(total_borrowed, offer_rate, offer_term, term)

    # Compute monthly payment according to formula here:
    remaining_term = (term - offer_term)
    m_payments = calc_monthly_payment(balance, interest_rate, remaining_term)
    later_payments = np.array([m_payments] * remaining_term * 12)

    # Generate plot data
    x = np.array(range(1, term * 12 + 1))
    y = np.append(initial_payments, later_payments)

    # Create figure dict
    figure = {
        'data': [
            {'x': x,
             'y': y,
             'type': 'bar',
             },
        ],
        'layout': {
            'clickmode': 'event+select'
        }
    }
    interest_paid = int(np.sum(y) - total_borrowed)
    total = f"Total interest paid: £{interest_paid:,}"
    return figure, total


def calc_monthly_payment(total_borrowed, r, term):
    """
    Function to compute monthly mortgage repayments.
    https://en.wikipedia.org/wiki/Mortgage_calculator

    Args:
        total_borrowed (): total amount borrowed
        r (): annual interest rate (%)
        term (): length of mortgage (years)
    """
    n_payments = term * 12
    offer_r = (r / 12) / 100
    return (total_borrowed * offer_r) / (1 - (1 + offer_r) ** -n_payments)

def compute_remaining_balance(total_borrowed, r, offer_term, term):
    """
    Computes remaining balance left on loan after end of offer_term periods

    Args:
        total_borrowed (): total amount initially borrowed
        r (): interest rate paid (%)
        offer_term (): length of offer period
        term (): total length of loan

    Returns:
        remaining balance on loan
    """
    # Formula from https://www.mtgprofessor.com/formulas.htm
    offer_r = (r / 12) / 100
    offer_months = offer_term * 12
    n_payments = term * 12
    return total_borrowed * ((1 + offer_r) ** n_payments - (1 + offer_r) ** offer_months) / (
                (1 + offer_r) ** n_payments - 1)


if __name__ == '__main__':
    app.run_server(debug=True)
