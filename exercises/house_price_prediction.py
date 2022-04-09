import os.path
import sys

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy import stats
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    houses_df = pd.read_csv(filename)

    houses_df['date'] = pd.to_datetime(houses_df['date'].str[:-7])
    houses_df['view'] = houses_df['view'] + 1

    for col in houses_df.columns[2:]:
        if col not in ["zipcode", "lat", "long"]:
            houses_df[col][houses_df[col] < 0] = np.nan

    for col in houses_df.columns[1:]:
        if col not in ['waterfront', 'sqft_basement', 'yr_renovated']:
            houses_df[col][houses_df[col] == 0] = np.nan

    houses_df.dropna(inplace=True)

    houses_df['year_of_sale'] = houses_df['date'].dt.year
    houses_df['month_of_sale'] = houses_df['date'].dt.month
    houses_df['day_of_sale'] = houses_df['date'].dt.day

    houses_df = houses_df.drop(labels=['date'], axis=1)
    houses_df = houses_df[(np.abs(stats.zscore(houses_df)) < 3).all(axis=1)]
    houses_df['built_sale_gap'] = houses_df['year_of_sale'] - houses_df[
        'yr_built']
    houses_df['reno_sale_gap'] = houses_df['year_of_sale'] - houses_df[
        'yr_renovated']
    houses_df['reno_sale_gap'] = np.where(houses_df['reno_sale_gap'] > houses_df['built_sale_gap'].max(), houses_df['built_sale_gap'].max(), houses_df['reno_sale_gap'])
    houses_df['was_renovated'] = np.where(houses_df['yr_renovated'] > 0, 1, 0)
    houses_df['has_basement'] = np.where(houses_df['sqft_basement'] > 0, 1, 0)
    houses_df['years_since_work'] = houses_df[
        ['built_sale_gap', 'reno_sale_gap']].min(axis=1)
    houses_df['living_compare'] = houses_df['sqft_living'] / houses_df[
        'sqft_living15'] * 100
    houses_df['lot_compare'] = houses_df['sqft_lot'] / houses_df[
        'sqft_lot15'] * 100

    houses_df.reset_index(inplace=True)
    price_array = houses_df['price']
    houses_df = houses_df[["year_of_sale", "month_of_sale", "day_of_sale",
                           "bedrooms", "bathrooms", "floors", "sqft_living",
                           "sqft_lot", "sqft_living15", "sqft_lot15",
                           "living_compare", "lot_compare", "yr_built",
                           "yr_renovated", "built_sale_gap", "reno_sale_gap",
                           "years_since_work", "waterfront", "view",
                           "condition", "grade", "sqft_above", "sqft_basement",
                           "was_renovated", "has_basement"]]
    price_array = price_array.loc[houses_df.index.to_list()]

    return houses_df, price_array


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def calculate_corr(y, x):
        return np.cov(x, y)[0][1] / (np.std(x)*np.std(y))

    for col in X.columns:
        corr = calculate_corr(y, X[col])
        title_str = "Correlation Between " + col + " and the Price is: " + str(corr)
        fig = px.scatter(x=X[col],
                         y=y,
                         trendline='ols',
                         title=title_str,
                         trendline_color_override="red")
        fig.update_xaxes(title_text=col)
        fig.update_yaxes(title_text="Price")

        fig.write_image(os.path.join(output_path, col +".png"))


if __name__ == '__main__':

    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, y = load_data("/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/"
                      "IML.HUJI/datasets/house_prices.csv")

    # print(df.index.nunique() == len(df.index))
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, y, "/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/Exercises /ex2/images")

    # Question 3 - Split samples into training- and testing sets.
    from sklearn.model_selection import train_test_split
    train_df, train_y, test_df, test_y = split_train_test(df, y, .75)


    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    loss_for_percentage = {}
    for p in range(10, 101):
        losses = []
        for i in range(10):
            sample_train_df = train_df.sample(frac=(p/100))
            sample_train_y = train_y.loc[sample_train_df.index]
            linear_regression = LinearRegression().fit(sample_train_df.to_numpy(),
                                                       sample_train_y.to_numpy())
            loss = linear_regression.loss(test_df.to_numpy(), test_y.to_numpy())
            losses.append(loss)
        loss_for_percentage[p] = (np.array(losses).mean(), np.array(losses).std())

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss_df = pd.DataFrame.from_dict(loss_for_percentage, orient='index', columns=['mean', 'std'])
    loss_df['upper_limit'] = loss_df['mean'] + 2*(loss_df['std'])
    loss_df['lower_limit'] = loss_df['mean'] - 2 *(loss_df['std'])

    fig = go.Figure([
        go.Scatter(
            x=loss_df.index,
            y=loss_df['mean'],
            line=dict(dash="dash"),
            mode='markers+lines',
            name="Mean Loss",
            marker=dict(color="green", opacity=.7)
        ),
        go.Scatter(
            x=loss_df.index,
            y=loss_df.lower_limit,
            fill=None, mode="lines",
            line=dict(color="lightgrey"), showlegend=False
        ),
        go.Scatter(x=loss_df.index, y=loss_df.upper_limit, fill='tonexty',
                   mode="lines", line=dict(color="lightgrey"),
                   showlegend=False)],
        layout=go.Layout(
            title_text="Mean Loss as Function of the "
                                 "percentage of samples trained",
            xaxis={"title": "Percentage of train samples used"},
            yaxis={"title": "Mean Loss"}))

    fig.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                    "Semester_B/67577_IML/Exercises /ex2/images/mean_loss.png")


