import os
import sys
import scipy.stats
import statsmodels.api as sm
import bisect
from collections import defaultdict
import math
import numpy as np
import statistics
import itertools
import pandas as pd
import plotly.express as px
import statsmodels
import statsmodels.api as sm
from plotly import graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import scipy.stats
import warnings
import numpy
import pandas
from scipy import stats


# *************** Defining all necessary functions ******************


def onehotencoder(df, df_cat):
    onehotencoder = OneHotEncoder(handle_unknown="ignore")
    encoder = onehotencoder.fit_transform(df_cat.values.reshape(-1, 1)).toarray()
    dfOneHot = pd.DataFrame(encoder)
    data = pd.concat([df.select_dtypes(exclude=["object"]), dfOneHot], axis=1)
    data = data.head(len(df))
    return data


def boundaries(X):
    stdev = statistics.stdev(X)
    bin_width = 3.49 * (stdev) * (len(X)) ** (-(1 / 3))
    bin_number = round(math.sqrt(len(X)))
    sorted_pred = sorted(X)

    boundaries = []
    for i in range(0, bin_number):
        boundaries.append(sorted_pred[0] + bin_width * i)
    return boundaries


def MeanSquaredDiff(X):
    bn = boundaries(X)
    pop_mean = statistics.mean(X)
    dic = defaultdict(list)
    total_population = len(X)

    for x in X:
        ind = bisect.bisect_right(bn, x)
        dic[ind].append(x)
    list_df = list(dic.values())

    for j in range(0, len(list_df) - 1):
        chunk_mean = statistics.mean(list_df[j])
        msf = (chunk_mean - pop_mean) ** 2
    MeanSquaredDiff = msf.sum() / total_population
    return MeanSquaredDiff


def bin_average(X):
    bn = boundaries(X)
    dic = defaultdict(list)
    for x in X:
        ind = bisect.bisect_right(bn, x)
        dic[ind].append(x)
    list_df = list(dic.values())

    mean = []
    for j in range(0, len(list_df) - 1):
        chunk_mean = statistics.mean(list_df[j])
        mean.append(chunk_mean)
    return mean


def WeightedMeanSquaredDiff(X):
    bn = boundaries(X)
    pop_mean = statistics.mean(X)
    dic = defaultdict(list)
    total_population = len(X)

    for x in X:
        ind = bisect.bisect_right(bn, x)
        dic[ind].append(x)
    list_df = list(dic.values())

    for j in range(0, len(list_df) - 1):
        chunk_mean = statistics.mean(list_df[j])
        PopulationProportion = len(list_df[j]) / total_population
        msf = (chunk_mean - pop_mean) ** 2
    weightedMeanSquaredDiff = (PopulationProportion * msf.sum()) / total_population
    return weightedMeanSquaredDiff


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    corr_coeff = numpy.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pandas.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = numpy.sqrt(
                    phi2_corrected / numpy.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = numpy.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = numpy.sqrt(phi2 / numpy.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = numpy.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation(categories, values):
    f_cat, _ = pandas.factorize(categories)
    cat_num = numpy.max(f_cat) + 1
    y_avg_array = numpy.zeros(cat_num)
    n_array = numpy.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[numpy.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = numpy.average(cat_measures)
    y_total_avg = numpy.sum(numpy.multiply(y_avg_array, n_array)) / numpy.sum(n_array)
    numerator = numpy.sum(
        numpy.multiply(
            n_array, numpy.power(numpy.subtract(y_avg_array, y_total_avg), 2)
        )
    )
    denominator = numpy.sum(numpy.power(numpy.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numpy.sqrt(numerator / denominator)
    return eta


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return numpy.array([value if value is not None else 0 for value in data])


def variable_cat_plot(x, y, path=None):
    conf_matrix = confusion_matrix(x, y)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title=x.name,
        yaxis_title=y.name,
    )
    if path is not None:
        fig_no_relationship.write_html(path, include_plotlyjs="cdn")
    else:
        fig_no_relationship.show()
    return


def variable_con_plot(x, y, path=None):
    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Two Continuous Predictors",
        xaxis_title=x.name,
        yaxis_title=y.name,
    )
    if path is not None:
        fig.write_html(path)
    else:
        fig.show()
    return


def hist(x_label, y_label, path=None):
    df = px.data.tips()
    fig = px.histogram(df, x=x_label, y=y_label, color=x_label, marginal="rug",
                       )
    fig.update_layout(
        xaxis_title=x_label.name,
        yaxis_title=y_label.name,
        legend_title=x_label.name,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    if path is not None:
        fig.write_html(path)
    else:
        fig.show()
    return


def violin(x_label, y_label, path=None):
    df = px.data.tips()
    fig = px.violin(df, x=x_label, y=y_label, color=x_label, box=True, points="all",
                    )
    fig.update_layout(
        xaxis_title=x_label.name,
        yaxis_title=y_label.name,
        legend_title=x_label.name,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    if path is not None:
        fig.write_html(path)
    else:
        fig.show()
    return

def dataframe_to_html(df, hyperlink_columns, out_path):
    def make_hyperlink(path):
        f_url = os.path.basename(path)
        return u'<a href="{}">{}</a>'.format(path, f_url)

    # This css class helps to make our table look stylish
    css_style = '<link rel="stylesheet" href="https://cdn.jupyter.org/notebook/5.1.0/style/style.min.css">'
    # Set formatter for the hyperlink columns
    formatters = {}
    for hyperlink_column in hyperlink_columns:
        formatters[hyperlink_column] = make_hyperlink
    # Generate HTML table from dataframe
    html_output = df.to_html(classes="rendered_html", formatters=formatters, justify="center", escape=False)

    # Write HTML output into file along with a link to our css style
    with open(out_path, "w") as out_file:
        # Writing data to a file
        out_file.write(css_style)
        out_file.write(html_output)


def init_directories():
    try:
        os.mkdir("concat_plot")
        os.mkdir("cat_plot")
        os.mkdir("con_plot")
        os.mkdir("catcon_plot")
    except:
        pass


def main():
    global table_final
    init_directories()

    # *************** Reading Dataset ******************
    # Explanation 1: For the sake of running time, I'm doing all the analysis on the first 100 rows of dataset.
    # You can run whole code with changing df_full to df and removing line 13.

    # Explanation 2: I deleted all columns with only 1 unique values, since they cannot contribute to the model.

    df_full = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    )

    df = df_full.head(100)

    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)

    print(df.to_string())

    # *************** Identifying Response and Predictors and Their Type ******************

    responses = ["traffic_volume"]
    predictors = ["temp", "clouds_all", "weather_main", "weather_description"]

    response_type = ""

    for i in responses:
        if df[i].nunique() == 2:
            df[i] = df[i].astype("bool")
            df.replace({False: 0, True: 1}, inplace=True)
            response_type = "categorical"
        else:
            response_type = "continuous"

    predictors_type = {"continuous": [], "categorical": []}
    continuous = df.select_dtypes(include=["float", "int"])
    for i in predictors:
        if i in list(continuous) and df[i].nunique() > 5:
            predictors_type["continuous"].append(i)
        else:
            predictors_type["categorical"].append(i)

    print("Response variable is:", *responses)
    print("Response type is:", response_type)

    print("Predictor variables are:", predictors)
    print("Predictors types:", predictors_type)

    # dividing dataframes to categorical and continuous

    for key, value in predictors_type.items():
        if key == "continuous":
            df_continuous = df[value]
        else:
            df_categorical = df[value]

    print(df_continuous)
    print(df_categorical)

    # creating list for continuous and categorical variables for iteration purposes

    predictors_con = []
    predictors_cat = []
    for i in predictors:
        if i in df_continuous:
            predictors_con.append(i)
        else:
            predictors_cat.append(i)

    # *************** Handling Null Values ******************

    for col in df.columns:
        if (
                df[col].dtypes == "float"
                or df[col].dtypes == "int"
                and df[col].nunique() > 5
        ):
            df[col].fillna((df[col].mean()), inplace=True)
        else:
            df = df.apply(lambda col: col.fillna(col.value_counts().index[0]))

    # *************** One Hot Encoder ******************

    data_cat = df.select_dtypes("object")
    data = onehotencoder(df, data_cat)

    # *************** Test and Train Datasets ******************

    for i in responses:
        x = data.drop(i, axis=1)
        y = data[i]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    for i in responses:
        if response_type == "categorical":
            logr = LogisticRegression()
            logr_fitted = logr.fit(x_train, y_train)
            logr_predict = logr_fitted.predict(x_test)
            print(logr_predict)
            print(logr_fitted.summary())
        else:
            feature_name = i
            ols_predict = statsmodels.api.add_constant(x)
            ols = statsmodels.api.OLS(y, ols_predict)
            ols_fitted = ols.fit()
            predictor_ols = ols_fitted.predict()
            print(predictor_ols)
            print(f"Variable: {feature_name}")
            print(ols_fitted.summary())

    # *************** Correlation Tables for all 3 possibilities ******************

    # 1. creating permutations:

    combo = set(itertools.combinations(predictors, 2))

    # 2. creating the tables:

    table_con = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Pearson Correlation",
            "Absolute Value of Correlation",
            "Linear Regression Plot",
        ]
    )

    table_cat = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Cramers V",
            "Absolute Value of Correlation",
            "heatmap"
        ]
    )

    table_catcon = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Correlation ratio",
            "Absolute Value of Correlation",
            "Violin Plot",
            "Histogram Plot"
        ]
    )

    table_concat = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Correlation ratio",
            "Absolute Value of Correlation",
            "Violin Plot",
            "Histogram Plot"
        ]
    )

    # 3. Fill in the Correlation tables and draw plots
    # Explanation: First it calculate correlation and add it to the proper table, then it creates plots.

    for index, tup in enumerate(combo):
        if tup[0] in predictors_con and tup[1] in predictors_con:
            x_label = df[tup[0]]
            y_label = df[tup[1]]
            pearson = scipy.stats.pearsonr(x_label, y_label).statistic
            path = "{}/{}_{}_{}".format("con_plot", tup[0], tup[1], "con_plot.html")
            variable_con_plot(x_label, y_label, path)
            new = [x_label.name, y_label.name, pearson, np.abs(pearson), path]
            table_con.loc[len(table_con)] = new
        elif tup[0] in predictors_cat and tup[1] in predictors_cat:
            x_label = df[tup[0]]
            y_label = df[tup[1]]
            fill_na(df)
            correlation = cat_correlation(x_label, y_label)
            path = "{}/{}_{}_{}".format("cat_plot", tup[0], tup[1], "cat_plot.html")
            variable_cat_plot(x_label, y_label, path)
            new = [x_label.name, y_label.name, correlation, np.abs(correlation), path]
            table_cat.loc[len(table_cat)] = new
        elif tup[0] in predictors_cat and tup[1] in predictors_con:
            x_label = df[tup[0]]
            y_label = df[tup[1]]
            correlation = cat_cont_correlation(x_label, y_label)
            path1 = "{}/{}_{}_{}".format("catcon_plot", tup[0], tup[1], "catcon_plot1.html")
            path2 = "{}/{}_{}_{}".format("catcon_plot", tup[0], tup[1], "catcon_plot2.html")
            hist(x_label, y_label, path1)
            violin(x_label, y_label, path2)
            new = [x_label.name, y_label.name, correlation, np.abs(correlation), path1, path2]
            table_catcon.loc[len(table_catcon)] = new
        elif tup[0] in predictors_con and tup[1] in predictors_cat:
            x_label = df[tup[1]]
            y_label = df[tup[0]]
            correlation = cat_cont_correlation(x_label, y_label)
            path1 = "{}/{}_{}_{}".format("concat_plot", tup[0], tup[1], "concat_plot1.html")
            path2 = "{}/{}_{}_{}".format("concat_plot", tup[0], tup[1], "concat_plot2.html")
            violin(x_label, y_label, path2)
            hist(x_label, y_label, path1)
            new = [x_label.name, y_label.name, correlation, np.abs(correlation), path1, path2]
            table_concat.loc[len(table_concat)] = new
            table_final = pd.concat([table_concat, table_catcon])

    dataframe_to_html(table_cat, ["heatmap"], "tablecat.html")
    dataframe_to_html(table_con, ["Linear Regression Plot"], "tablecon.html")
    dataframe_to_html(table_final, ["Violin Plot", "Histogram Plot"], "tableboth.html")
    # *************** Brute Force for all 3 possibilities ******************

    # 1. creating the tables:

    brute_force_con = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
        ]
    )

    brute_force_cat = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
        ]
    )

    brute_force_both1 = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
        ]
    )

    brute_force_both2 = pd.DataFrame(
        columns=[
            "predictor 1",
            "predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
        ]
    )

    # 2. Fill in the Brute Force tables

    for i in responses:
        if response_type == "continuous":
            for index, tup in enumerate(combo):
                if tup[0] in predictors_con and tup[1] in predictors_con:
                    x = tup[0]
                    y = tup[1]
                    dataset = data[[x, y]]
                    model = sm.OLS(df[i], dataset, axis=1).fit()
                    pred = model.predict()
                    meansquareddiff = MeanSquaredDiff(pred)
                    weightedmeansquareddiff = WeightedMeanSquaredDiff(pred)
                    # corr = model.rsquared ** .5
                    con_new = [x, y, meansquareddiff, weightedmeansquareddiff]
                    brute_force_con.loc[len(brute_force_con)] = con_new
                elif tup[0] in predictors_cat and tup[1] in predictors_cat:
                    x = tup[0]
                    y = tup[1]
                    dataset = df[[x, y]]
                    dt = onehotencoder(dataset, dataset)
                    model = sm.OLS(df[i], dt, axis=1).fit()
                    pred = model.predict()
                    # corr = model.rsquared ** .5
                    meansquareddiff = MeanSquaredDiff(pred)
                    weightedmeansquareddiff = WeightedMeanSquaredDiff(pred)
                    cat_new = [x, y, meansquareddiff, weightedmeansquareddiff]
                    brute_force_cat.loc[len(brute_force_cat)] = cat_new
                elif tup[0] in predictors_cat and tup[1] in predictors_con:
                    x = tup[0]
                    y = tup[1]
                    dataset = df[[x, y]]
                    dt = onehotencoder(dataset, df[x])
                    model = sm.OLS(df[i], dt, axis=1).fit()
                    pred = model.predict()
                    meansquareddiff = MeanSquaredDiff(pred)
                    weightedmeansquareddiff = WeightedMeanSquaredDiff(pred)
                    both1 = [x, y, meansquareddiff, weightedmeansquareddiff]
                    brute_force_both1.loc[len(brute_force_both1)] = both1
                elif tup[0] in predictors_con and tup[1] in predictors_cat:
                    x = tup[0]
                    y = tup[1]
                    dataset = df[[x, y]]
                    dt = onehotencoder(dataset, df[y])
                    model = sm.OLS(df[i], dt, axis=1).fit()
                    pred = model.predict()
                    meansquareddiff = MeanSquaredDiff(pred)
                    weightedmeansquareddiff = WeightedMeanSquaredDiff(pred)

    both2 = [x, y, meansquareddiff, weightedmeansquareddiff]
    brute_force_both2.loc[len(brute_force_both2)] = both2
    brute_force_both = pd.concat([brute_force_both1, brute_force_both2])

    # *************** heatmap for all 3 possibilities ******************

    # sns.heatmap(df_continuous)
    # plt.show()

    # df_cat = df_categorical.apply(lambda x: pd.factorize(x)[0]).corr(method="pearson", min_periods=1)
    # sns.heatmap(df_cat, annot=True)
    # plt.show()

    # df_all = df.apply(lambda x: pd.factorize(x)[0]).corr(method="pearson", min_periods=1)
    # sns.heatmap(df_all, annot=True)
    # plt.show()

    # dataframe_to_html(brute_force_con, ["Residual Plot"], "brutetablcon.html")
    # *************** Printing Tables for Each Possibility ******************

    # # 1. both categorical
    print(table_cat.sort_values(["Cramers V"], ascending=[False]).to_string())
    print(brute_force_cat.to_string())

    # # 2. both continuous
    print(table_con.sort_values(["Pearson Correlation"], ascending=[False]).to_string())
    print(brute_force_con.to_string())

    # # 3. categorical and continuous
    print(table_final.sort_values(["Correlation ratio"], ascending=[False]).to_string())
    print(brute_force_both.to_string())


    return


if __name__ == "__main__":
    sys.exit(main())
