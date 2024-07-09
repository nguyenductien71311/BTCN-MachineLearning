import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# from module import lstml_model

rcParams["figure.figsize"] = 20, 10
scaler = MinMaxScaler(feature_range=(0, 1))
currencies = ["BTC", "ETH", "ADA"]

def analyze_dataset(df):
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    numeric_columns = ["Open", "High", "Low", "Close"]
    
    for column in numeric_columns:
        df[column] = df[column].astype(str).str.replace(",", "").astype(float)

    df.index = df["Date"]

    plt.figure(figsize=(16, 8))
    plt.plot(df["Close"], label="Close Price history")

    return df

def sort_dataset(df):
    data = df.sort_index(ascending=True, axis=0)

    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=["Date", "Close"])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data["Date"][i]
        new_dataset["Close"][i] = data["Close"][i]

    return new_dataset

def normalize_dataset(new_dataset):
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    final_dataset = new_dataset.values

    train_data = final_dataset[0:987, :]
    valid_data = final_dataset[987:, :]

    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(scaled_data[i - 60 : i, 0])
        y_train_data.append(scaled_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(
        x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1)
    )

    return x_train_data, y_train_data, valid_data

def test_dataset(inputs_data, traning_model):
    X_test = []

    for i in range(60, inputs_data.shape[0]):
        X_test.append(inputs_data[i - 60 : i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_closing_price = traning_model.predict(X_test)

    predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

def visualize_dataset(new_dataset, predicted_closing_price):
    train_data = new_dataset[:987]
    valid_data = new_dataset[987:]

    valid_data["Predictions"] = predicted_closing_price

    plt.plot(train_data["Close"], label="Train Data")
    plt.plot(valid_data[["Close", "Predictions"]], label="Validation Data")

def lstml_model(x_train_data, y_train_data):
    lstm_model = Sequential()

    lstm_model.add(
        LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1))
    )

    lstm_model.add(LSTM(units=50))

    lstm_model.add(Dense(1))

    lstm_model.compile(loss="mean_squared_error", optimizer="adam")

    lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

    return lstm_model

def training_model(name):
    df = pd.read_csv(f"./dataset/{name}-USD.csv")

    df = analyze_dataset(df)

    new_dataset = sort_dataset(df)

    x_train_data, y_train_data, valid_data = normalize_dataset(new_dataset)

    traning_model = lstml_model(x_train_data, y_train_data)

    inputs_data = new_dataset[len(new_dataset) - len(valid_data) - 60 :].values
    inputs_data = inputs_data.reshape(-1, 1)  
    inputs_data = scaler.transform(inputs_data)

    predicted_closing_price = test_dataset(inputs_data, traning_model)

    traning_model.save(f"./model/{name.lower()}_usd.h5")
    
    visualize_dataset(new_dataset, predicted_closing_price)

    print(f"Train {name} currency successfully!")
    print("==================================================")
    print(" ")

for currency in currencies:
    training_model(currency)

print("Train model done!")
