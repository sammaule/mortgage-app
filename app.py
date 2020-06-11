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
        dcc.Input(id='offer-term', value=2, type='number'),
    ]),
    html.Div([
        "Mortgage term (years): ",
        dcc.Input(id='mortgage-term', value=20, type='number'),
    ]),
    html.Div([
        "Initial interest rate (%): ",
        dcc.Input(id='initial-interest-rate', value=1.0, type='number'),
    ]),
    html.Div([
        "Interest rate (%): ",
        dcc.Input(id='interest-rate', value=2.0, type='number'),
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
    # TODO: write helper functions to simplify this callback one for computing monthly payments
    # One for computing remaining balance

    # Compute initial monthly payment
    total_borrowed = total * 1000
    n_payments = term * 12
    offer_months = offer_term * 12
    offer_r = (offer_rate / 12) / 100
    offer_m_payment = (total_borrowed * offer_r) / (1 - (1 + offer_r)**-n_payments)
    initial_payments = np.array([offer_m_payment] * offer_months)

    # remaining balance on loan
    # Formula from https://www.mtgprofessor.com/formulas.htm
    balance = total_borrowed * ((1 + offer_r)**n_payments - (1 + offer_r)**offer_months) / ((1 + offer_r)**n_payments - 1)

    # Compute monthly payment according to formula here:
    # https://en.wikipedia.org/wiki/Mortgage_calculator

    n_payments = (term - offer_term) * 12
    monthly_r = (interest_rate / 12) / 100
    monthly_payment = (balance * monthly_r) / (1 - (1 + monthly_r)**-n_payments)
    later_payments = np.array([monthly_payment] * n_payments)

    # Generate plot data
    x = np.array(range(1, n_payments + 1))
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


if __name__ == '__main__':
    app.run_server(debug=True)
