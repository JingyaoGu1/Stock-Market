from flask import Flask, jsonify, request
import numpy as np
from datetime import datetime
import yfinance as yf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Prediction function as defined earlier
def predict_stock_price(stock_symbol):
    yf.pdr_override()
    df = pdr.get_data_yahoo(stock_symbol, start='2018-01-01', end=datetime.now())

    # Create a new dataframe with only the 'Close column 
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    # Scale the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    last_60_days = scaled_data[-60:]
    # Reshape the data to be 3D (samples, time steps, features)
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    # Get the predicted scaled price
    predicted_price_scaled = model.predict(last_60_days)

    # Undo the scaling
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    print("Predicted Next Day Price: ", predicted_price[0][0])
    return float(predicted_price[0][0])


# Flask route for predictions
@app.route('/predict/<string:stock_symbol>', methods=['GET'])
def predict(stock_symbol):
    try:
        predicted_price = predict_stock_price(stock_symbol)
        return jsonify({'predictedPrice': predicted_price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False in a production environment
