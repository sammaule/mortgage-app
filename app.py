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
        "Mortgage term: ",
        dcc.Input(id='mortgage-term', value=20, type='number'),
    ]),
    html.Div([
        "Interest rate (%): ",
        dcc.Input(id='interest-rate', value=2.0, type='number'),
    ]),
    html.Div(id="total-repaid"
             ),
])


@app.callback(
    [Output('mortgage-plot', 'figure'),
     Output('total-repaid', 'children')
     ],
    [Input('mortgage-size', 'value'),
     Input('mortgage-term', 'value'),
     Input('interest-rate', 'value')]
)
def plot_monthly_repayments(total, term, interest_rate):

    # Compute monthly payment according to formula here:
    # https://en.wikipedia.org/wiki/Mortgage_calculator
    total_borrowed = total * 1000
    n_payments = term * 12
    monthly_r = (interest_rate / 12) / 100
    monthly_payment = (total_borrowed * monthly_r) / (1 - (1 + monthly_r)**-n_payments)

    # Generate plot data
    x = np.array(range(1, n_payments + 1))
    y = np.array([monthly_payment] * n_payments)

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
