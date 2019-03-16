import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import csv

style.use('ggplot')


# DATA PREPARATION
def get_data(filename, forecast_feature, test_size):
    dates = []
    prices = []
    with open(filename, 'r') as csvfile:
        csvfr = csv.reader(csvfile)
        header = next(csvfr)
        feature_col = header.index(forecast_feature)
        for row in csvfr:
            dates.append(int(''.join(row[0].split('-'))))  # remove hyphens in date(2018-01-05) string and convert to int
            prices.append(float(row[feature_col]))  # take forecast column data

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(prices, (len(prices), 1))

    x_train, x_test, y_train, y_test = train_test_split(dates, prices, test_size=test_size)
    return x_train,x_test, y_train, y_test


# Linear Regression model
def train_model(X_train, Y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)
    return regressor


def visualize(regressor, dates, prices):
    plt.scatter(dates, prices, color='yellow', label= 'Actual Price') #plotting the initial datapoints
    plt.plot(dates, regressor.predict(dates), color='red', linewidth=3, label = 'Predicted Price') #plotting the line made by linear regression
    plt.title('Linear Regression | Time vs. Price')
    plt.legend()
    plt.xlabel('Date Integer')
    plt.show()


# PREDICTION
def predict_price(regressor, datestr):
    date = int(datestr.replace("-", ""))
    date = [date]
    date = np.reshape(date, (len(date), 1))
    predicted_price =regressor.predict(date)
    print("Predicted Price on {} is {}".format(datestr, predicted_price))


def main():
    # Get Data of ICICBank Stock Prices from Yahoo Finance as CSV FILE for 10 Mar '18 to 10 Mar '19
    input_file_path = "data/raw/ICICIBANK.BO.csv"
    forecast_feature = "Open"   # choosing Open price column to forecast
    test_size = 0.2

    X_train, X_test, Y_train, Y_test = get_data(input_file_path, forecast_feature, test_size)

    regressor = train_model(X_train, Y_train)

    # visualize results with TRAIN dataset
    visualize(regressor, X_train, Y_train)
    # visualize results with TEST dataset
    visualize(regressor, X_test, Y_test)

    predict_price(regressor, "2019-02-12")

main()
