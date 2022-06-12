from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    y_ = (lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2))(x)
    epsilon = np.random.normal(0, noise, len(x))
    y = y_ + epsilon
    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), (2/3.0))
    train_x, train_y, test_x, test_y = train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()

    fig = go.Figure([go.Scatter(x=x, y=y_, name="Noiseless data", mode= "markers",
                                showlegend=True,
                                marker=dict(color="black", opacity=.7),
                                line=dict(color="black", dash="dash",
                                          width=1))],
                    layout=go.Layout(title=f"Simulated Polynomial with Train and Test, noise = {noise}, n_samples = {n_samples}",
                                     xaxis={
                                         "title": "x - Explanatory Variable"},
                                     yaxis={"title": "y - Response"},
                                     height=400))
    fig.add_trace(go.Scatter(x=train_x.reshape(-1,), y=train_y, name="Train data", mode="markers",
                             marker=dict(color="blue", opacity=.7), line=dict(width=1)))
    fig.add_trace(go.Scatter(x=test_x.reshape(-1,), y=test_y, name="Test data", mode="markers",
                             marker=dict(color="green", opacity=.7), line=dict(width=1)))
    fig.show()
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex5/figures/scatter_{n_samples}_{noise}.png")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, val_errors = [], []
    k_range = list(range(11))
    for k in k_range:
        p_fit = PolynomialFitting(k)
        train_loss, val_loss = cross_validate(p_fit, train_x, train_y, mean_square_error)
        train_errors.append(train_loss)
        val_errors.append(val_loss)

    fig = go.Figure([
        go.Scatter(name='Train Error', x=k_range, y=train_errors,
                   mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Mean Validation Error', x=k_range, y=val_errors,
                   mode='markers+lines', marker_color='rgb(220,179,144)')
    ])

    fig.update_layout(
        title=f"Train and Validation loss per K value, noise = {noise}, n_samples = {n_samples}",
        xaxis_title=r"$k\text{ - Polynomial degree}$",
        yaxis_title=r"$\text{Error Value}$"
    )

    fig.show()
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex5/figures/loss_per_k_{n_samples}_{noise}.png")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_hat = np.argmin(val_errors)
    p_fit = PolynomialFitting(k_hat).fit(train_x, train_y)
    print("Best k value is: ", k_hat)
    test_loss = mean_square_error(test_y, p_fit.predict(test_x))
    print("Test loss is: ", np.round(test_loss, 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y, (n_samples/len(X)))
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # for model in [RidgeRegression, Lasso]:
    ridge_lambda = regularization_exploration(train_X, train_y, RidgeRegression, np.linspace(0, 2, n_evaluations), "Ridge")
    lasso_lambda = regularization_exploration(train_X, train_y, Lasso,
                               np.linspace(0, 4, n_evaluations), "Lasso")

    ridge_model = RidgeRegression(ridge_lambda).fit(train_X, train_y)
    lasso_model = Lasso(lasso_lambda).fit(train_X, train_y)
    least_squares_model = LinearRegression().fit(train_X, train_y)
    print("=====================================================")
    print(f"ridge model loss is: {mean_square_error(test_y, ridge_model.predict(test_X))}")
    print(f"lasso model loss is: {mean_square_error(test_y, lasso_model.predict(test_X))}")
    print(f"Least squares model loss is: {mean_square_error(test_y, least_squares_model.predict(test_X))}")
    print("=====================================================")


def regularization_exploration(X, y, model, lambda_array, name):
    train_errors, val_errors = [], []
    for lam in lambda_array:
        estimator = model(lam)
        train_loss, val_loss = cross_validate(estimator, X, y,
                                              mean_square_error)
        train_errors.append(train_loss)
        val_errors.append(val_loss)

    lambda_hat = lambda_array[np.argmin(val_errors)]
    fig = go.Figure([
        go.Scatter(name='Train Error', x=lambda_array, y=train_errors,
                   mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Mean Validation Error', x=lambda_array, y=val_errors,
                   mode='markers+lines', marker_color='rgb(220,179,144)')
    ])
    fig.update_layout(
        title=f"Train and Validation loss per lambda value {name}, lowest lambda = {lambda_hat}",
        xaxis_title=r"$k\text{ - lambda value}$",
        yaxis_title=r"$\text{Error Value}$"
    )
    fig.show()
    fig.write_image(f"/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex5/figures/regression_loss_{name}.png")
    print(f"{name} model best lambda value is: {lambda_hat}")
    return lambda_hat

if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(noise=10, n_samples=1500)
    select_regularization_parameter()
