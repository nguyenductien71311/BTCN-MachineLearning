import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Read the CSV file
df = pd.read_csv("./dataset/stock_data.csv")
df_btc = pd.read_csv("./dataset/BTC-USD.csv", thousands=",")
df_eth = pd.read_csv("./dataset/ETH-USD.csv", thousands=",")
df_ada = pd.read_csv("./dataset/ADA-USD.csv", thousands=",")

# Load training model
model_btc = load_model("./model/btc_usd.h5")
model_eth = load_model("./model/eth_usd.h5")
model_ada = load_model("./model/ada_usd.h5")

# Initialize currency MinMaxScaler
scaler_btc = MinMaxScaler(feature_range=(0, 1))
scaler_eth = MinMaxScaler(feature_range=(0, 1))
scaler_ada = MinMaxScaler(feature_range=(0, 1))

# Convert the 'Date' column to datetime format for currency data
# Set the 'Date' column as the index for currency data
df_btc["Date"] = pd.to_datetime(df_btc.Date, format="%m/%d/%Y")
df_btc.index = df_btc["Date"]

df_eth["Date"] = pd.to_datetime(df_eth.Date, format="%m/%d/%Y")
df_eth.index = df_eth["Date"]

df_ada["Date"] = pd.to_datetime(df_ada.Date, format="%m/%d/%Y")
df_ada.index = df_ada["Date"]

# Sort currency data by 'Date' in ascending order
# Create a new DataFrame for currency with 'Date' and 'Close' columns
data_btc = df_btc.sort_index(ascending=True, axis=0)
new_data_btc = pd.DataFrame(index=range(0, len(df_btc)), columns=["Date", "Close"])

data_eth = df_eth.sort_index(ascending=True, axis=0)
new_data_eth = pd.DataFrame(index=range(0, len(df_eth)), columns=["Date", "Close"])

data_ada = df_ada.sort_index(ascending=True, axis=0)
new_data_ada = pd.DataFrame(index=range(0, len(df_ada)), columns=["Date", "Close"])

# Copy 'Date' and 'Close' data from data_currency to new_data_currency
for i in range(0, len(data_btc)):
    new_data_btc["Date"][i] = data_btc["Date"][i]
    new_data_btc["Close"][i] = data_btc["Close"][i]

for i in range(0, len(data_eth)):
    new_data_eth["Date"][i] = data_eth["Date"][i]
    new_data_eth["Close"][i] = data_eth["Close"][i]

for i in range(0, len(data_ada)):
    new_data_ada["Date"][i] = data_ada["Date"][i]
    new_data_ada["Close"][i] = data_ada["Close"][i]

# Set the 'Date' column as the index for new_data_currency
# Drop the 'Date' column from new_data_currency
new_data_btc.index = new_data_btc.Date
new_data_btc.drop("Date", axis=1, inplace=True)

new_data_eth.index = new_data_eth.Date
new_data_eth.drop("Date", axis=1, inplace=True)

new_data_ada.index = new_data_ada.Date
new_data_ada.drop("Date", axis=1, inplace=True)

# Convert the new DataFrame to a numpy array
dataset_btc = new_data_btc.values
dataset_eth = new_data_eth.values
dataset_ada = new_data_ada.values

# Split the data into training and validation sets
train_btc = dataset_btc[0:1000, :]
valid_btc = dataset_btc[1000:, :]

train_eth = dataset_eth[0:1000, :]
valid_eth = dataset_eth[1000:, :]

train_ada = dataset_ada[0:1000, :]
valid_ada = dataset_ada[1000:, :]

# Scale the data
scaled_data_btc = scaler_btc.fit_transform(dataset_btc)
scaled_data_eth = scaler_eth.fit_transform(dataset_eth)
scaled_data_ada = scaler_ada.fit_transform(dataset_ada)

# Select input data for currency by slicing new_data_currency based on the validation set length and a buffer of 96 additional data points
# Reshape the input data to have a single feature column
# Scale the input data using the currency scaler (scaler_btc)
inputs_btc = new_data_btc[len(new_data_btc) - len(valid_btc) - 96 :].values
inputs_btc = inputs_btc.reshape(-1, 1)
inputs_btc = scaler_btc.transform(inputs_btc)

inputs_eth = new_data_eth[len(new_data_eth) - len(valid_eth) - 95 :].values
inputs_eth = inputs_eth.reshape(-1, 1)
inputs_eth = scaler_eth.transform(inputs_eth)

inputs_ada = new_data_ada[len(new_data_ada) - len(valid_ada) - 96 :].values
inputs_ada = inputs_ada.reshape(-1, 1)
inputs_ada = scaler_ada.transform(inputs_ada)

# Initialize list to store test sequences
X_test_btc = []
X_test_eth = []
X_test_ada = []

# Iterate over the range starting from 96 to the number of rows in inputs_currency
# Append sequences of length 96 (from i - 96 to i) from the first column (index 0) of inputs_currency to X_test_currency
# Convert X_test_currency list to a NumPy array
for i in range(96, inputs_btc.shape[0]):
    X_test_btc.append(inputs_btc[i - 96 : i, 0])
X_test_btc = np.array(X_test_btc)

for i in range(95, inputs_eth.shape[0]):
    X_test_eth.append(inputs_eth[i - 95 : i, 0])
X_test_eth = np.array(X_test_eth)

for i in range(96, inputs_ada.shape[0]):
    X_test_ada.append(inputs_ada[i - 96 : i, 0])
X_test_ada = np.array(X_test_ada)

# Reshape X_test_currency to match the expected input shape of the model (number of samples, time steps, number of features)
# Use the trained model (model_currency) to predict closing prices for the reshaped X_test_currency
# Inverse transform (rescale) the predicted closing prices to their original scale using scaler_currency
X_test_btc = np.reshape(X_test_btc, (X_test_btc.shape[0], X_test_btc.shape[1], 1))
closing_price_btc = model_btc.predict(X_test_btc)
closing_price_btc = scaler_btc.inverse_transform(closing_price_btc)

X_test_eth = np.reshape(X_test_eth, (X_test_eth.shape[0], X_test_eth.shape[1], 1))
closing_price_eth = model_eth.predict(X_test_eth)
closing_price_eth = scaler_eth.inverse_transform(closing_price_eth)

X_test_ada = np.reshape(X_test_ada, (X_test_ada.shape[0], X_test_ada.shape[1], 1))
closing_price_ada = model_ada.predict(X_test_ada)
closing_price_ada = scaler_ada.inverse_transform(closing_price_ada)

# Select the first 1000 rows of new_data_currency for training
# Select rows from new_data_currency starting from index 1000 for validation
# Assign the predicted closing prices (closing_price_currency) to a new column "Predictions" in valid_currency
train_btc = new_data_btc[:1000]
valid_btc = new_data_btc[1000:]
valid_btc["Predictions"] = closing_price_btc

train_eth = new_data_eth[:1000]
valid_eth = new_data_eth[1000:]
valid_eth["Predictions"] = closing_price_eth

train_ada = new_data_ada[:1000]
valid_ada = new_data_ada[1000:]
valid_ada["Predictions"] = closing_price_ada

app = dash.Dash()
server = app.server

app.layout = html.Div(
    [
        html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            children=[
                dcc.Tab(
                    label="BTC-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Data BTC-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_btc.index,
                                                y=valid_btc["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Predicted Model Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Data BTC-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_btc.index,
                                                y=valid_btc["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="ETH-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Data ETH-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_eth.index,
                                                y=valid_eth["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Predicted Model Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Data ETH-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_eth.index,
                                                y=valid_eth["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="ADA-USD Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H2(
                                    "Actual Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Actual Data ADA-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_ada.index,
                                                y=valid_ada["Close"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                                html.H2(
                                    "Predicted Model Closing Price",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(
                                    id="Predicted Data ADA-USD",
                                    figure={
                                        "data": [
                                            go.Scatter(
                                                x=valid_ada.index,
                                                y=valid_ada["Predictions"],
                                                mode="markers",
                                            )
                                        ],
                                        "layout": go.Layout(
                                            title="scatter plot",
                                            xaxis={"title": "Date"},
                                            yaxis={"title": "Closing Rate"},
                                        ),
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Facebook Stock Data",
                    children=[
                        html.Div(
                            [
                                html.H1(
                                    "Stocks High vs Lows", style={"textAlign": "center"}
                                ),
                                dcc.Dropdown(
                                    id="my-dropdown",
                                    options=[
                                        {"label": "Tesla", "value": "TSLA"},
                                        {"label": "Apple", "value": "AAPL"},
                                        {"label": "Facebook", "value": "FB"},
                                        {"label": "Microsoft", "value": "MSFT"},
                                    ],
                                    multi=True,
                                    value=["FB"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="highlow"),
                                html.H1(
                                    "Stocks Market Volume",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Dropdown(
                                    id="my-dropdown2",
                                    options=[
                                        {"label": "Tesla", "value": "TSLA"},
                                        {"label": "Apple", "value": "AAPL"},
                                        {"label": "Facebook", "value": "FB"},
                                        {"label": "Microsoft", "value": "MSFT"},
                                    ],
                                    multi=True,
                                    value=["FB"],
                                    style={
                                        "display": "block",
                                        "margin-left": "auto",
                                        "margin-right": "auto",
                                        "width": "60%",
                                    },
                                ),
                                dcc.Graph(id="volume"),
                            ],
                            className="container",
                        ),
                    ],
                ),
            ],
        ),
    ]
)


@app.callback(Output("highlow", "figure"), [Input("my-dropdown", "value")])
def update_graph(selected_dropdown):
    dropdown = {
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "FB": "Facebook",
        "MSFT": "Microsoft",
    }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["High"],
                mode="lines",
                opacity=0.7,
                name=f"High {dropdown[stock]}",
                textposition="bottom center",
            )
        )
        trace2.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Low"],
                mode="lines",
                opacity=0.6,
                name=f"Low {dropdown[stock]}",
                textposition="bottom center",
            )
        )
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {
                                "count": 1,
                                "label": "1M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {
                                "count": 6,
                                "label": "6M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
            },
            yaxis={"title": "Price (USD)"},
        ),
    }
    return figure


@app.callback(Output("volume", "figure"), [Input("my-dropdown2", "value")])
def update_graph(selected_dropdown_value):
    dropdown = {
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "FB": "Facebook",
        "MSFT": "Microsoft",
    }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Volume"],
                mode="lines",
                opacity=0.7,
                name=f"Volume {dropdown[stock]}",
                textposition="bottom center",
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        "data": data,
        "layout": go.Layout(
            colorway=["#5E0DAC", "#FF4F00", "#375CB1", "#FF7400", "#FFF400", "#FF0056"],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={
                "title": "Date",
                "rangeselector": {
                    "buttons": list(
                        [
                            {
                                "count": 1,
                                "label": "1M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {
                                "count": 6,
                                "label": "6M",
                                "step": "month",
                                "stepmode": "backward",
                            },
                            {"step": "all"},
                        ]
                    )
                },
                "rangeslider": {"visible": True},
                "type": "date",
            },
            yaxis={"title": "Transactions Volume"},
        ),
    }
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)

print("Port model Successfully")
print("==================================================")
print(" ")
