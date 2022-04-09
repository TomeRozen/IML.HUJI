import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    temp_df = pd.read_csv(filename, parse_dates=['Date'])
    temp_df = temp_df[temp_df["Temp"] > -70]
    temp_df["DayOfYear"] = temp_df["Date"].dt.dayofyear
    temp_df["Year"] = temp_df["Year"].astype(str)
    temp_df = temp_df[["Country", "City", "Date", "Year", "Month", "Day",
                       "DayOfYear", "Temp"]]
    return temp_df

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temp_df = load_data("/Users/tomerrozenstine/Documents/Uni/bcs/year2/Semester_B/67577_IML/IML.HUJI/datasets/City_Temperature.csv")
    #
    # # Question 2 - Exploring data for specific country
    israel_temp_df = temp_df[temp_df["Country"] == "Israel"].reset_index()
    sctr_fig = px.scatter(israel_temp_df, x="DayOfYear", y="Temp",
                          color="Year",
                          title="Daily Temperature as Function"
                                " of the Day of The Year")
    sctr_fig.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                         "Semester_B/67577_IML/Exercises /ex2/poly_fit_plots/"
                         "Israel_temp_per_day.png")

    month_il = israel_temp_df.groupby("Month").agg(np.std)
    bar_plt = px.bar(month_il, y="Temp",
                     title="STD of the Daily Temp, as Function of Month")
    bar_plt.update_yaxes(title_text="Temp STD")
    bar_plt.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                         "Semester_B/67577_IML/Exercises /ex2/poly_fit_plots/"
                         "monthly_temp_std.png")

    # Question 3 - Exploring differences between countries
    country_month_df = temp_df.groupby(["Country", "Month"]).Temp.agg(
        ["mean", np.std])
    country_month_df_rev = country_month_df.reset_index("Country")
    line_plt = px.line(country_month_df_rev, y="mean", error_y="std",
                       color="Country",
                       title="Monthly Average Temprature, Divided by Country ")
    line_plt.update_yaxes(title_text="Average Temp")
    line_plt.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                         "Semester_B/67577_IML/Exercises /ex2/poly_fit_plots/"
                         "countries_line_plot.png")

    # Question 4 - Fitting model for different values of `k`
    israel_train_x, israel_train_y, israel_test_x, israel_test_y = \
        split_train_test(israel_temp_df[["DayOfYear"]], israel_temp_df["Temp"],
                         train_proportion=.75)

    loss_per_k = []
    for k in range(1, 11):
        poly_regression = PolynomialFitting(k).fit(israel_train_x.values,
                                                   israel_train_y.values)
        loss = np.round(poly_regression.loss(israel_test_x.values,
                                             israel_test_y.values), 2)
        loss_per_k.append(loss)
        print("Loss for k=", k, "is: ", loss)

    k_loss_bar = px.bar(x=range(1,11), y=loss_per_k,
                        title="Loss score per K value")
    k_loss_bar.update_yaxes(title_text="Loss")
    k_loss_bar.update_xaxes(title_text="K value")
    k_loss_bar.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                         "Semester_B/67577_IML/Exercises /ex2/poly_fit_plots/"
                         "k_val_loss_score.png")

    # Question 5 - Evaluating fitted model on different countries
    poly_regression = PolynomialFitting(5).fit(israel_temp_df[["DayOfYear"]].values,
                                               israel_temp_df["Temp"].values)
    loss_per_country = []
    for country in temp_df["Country"].unique():
        if country != "Israel":
            country_df = temp_df[temp_df["Country"] == country].reset_index()
            loss = poly_regression.loss(country_df.DayOfYear.values,
                                        country_df.Temp.values)

            loss_per_country.append((country,loss))

    loss_df = pd.DataFrame(loss_per_country,
                           columns=["Country", "Lost"])
    loss_bar_fig = px.bar(loss_df, x="Country", y="Lost",
                          title="Israel Fitted Temp Predection Model's Error, Per Country")
    loss_bar_fig.update_yaxes(title_text="Error")
    loss_bar_fig.write_image("/Users/tomerrozenstine/Documents/Uni/bcs/year2/"
                         "Semester_B/67577_IML/Exercises /ex2/poly_fit_plots/"
                         "loss_per_country.png")