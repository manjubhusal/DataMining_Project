from tkinter import Place

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Function to implement Gradient Boosting Regression on the dataset
def gradient_boosting_regression(x_train_regression, x_test_regression, y_train_regression, y_test_regression):
    print("\n---Gradient Boosting Regression----")
    start_time = datetime.now()
    gradient_regression = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_regression.fit(x_train_regression, y_train_regression)
    gb_predicted_y = gradient_regression.predict(x_test_regression)
    execution_time(start_time, "Gradient Boosting")
    regression_evaluation(y_test_regression, gb_predicted_y, "Gradient Boosting")

# Function to implement Polynomial Regression on the dataset
def polynomial_regression(x_train_regression, x_test_regression, y_train_regression, y_test_regression):
    print("\n---Polynomial Regression----")
    start_time = datetime.now()
    polynomial = PolynomialFeatures(degree = 2, include_bias = False)
    polynomial_x_train = polynomial.fit_transform(x_train_regression)
    polynomial_x_test = polynomial.transform(x_test_regression)
    regression_model = LinearRegression()
    regression_model.fit(polynomial_x_train,y_train_regression)
    polynomial_predicted_y = regression_model.predict(polynomial_x_test)
    execution_time(start_time, "Polynomial")
    regression_evaluation(y_test_regression, polynomial_predicted_y, "Polynomial")

# Function to evaluate the performance of regression models
def regression_evaluation(y_test_regression, predicted_y, model):
    # Calculating the Mean Squared Error and r^2 value in order to evaluate the model
    mse_value = mean_squared_error(y_test_regression, predicted_y)
    r2_value = r2_score(y_test_regression, predicted_y)
    print(f"Mean Squared Error(MSE) of {model} regression: {mse_value:.4f}")
    print(f"R-squared(R^2) of {model} regression: {r2_value:.4f}")

    # Plotting the actual and predicted values
    plt.figure(figsize=(10, 6))
    # Plotting KDE for actual values
    sns.kdeplot(y_test_regression, label="Actual Values", color="pink", linewidth=2)
    # Plotting KDE for predicted values
    sns.kdeplot(predicted_y, label="Predicted Values", color="purple", linewidth=2)
    # Plotting titles and labels to the graph
    plt.title(f"Actual versus Predicted Values of {model} Regression")
    plt.xlabel("BMXBMI")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def execution_time(start_time, model):
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    print(f"Total execution time for {model} regression is: ", total_time)