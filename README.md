# tempo-performance

Research on employee productivity pattern based on their tempo worklog analysis.

## Context

The main purpose of this project is to determine a **productivity pattern** of IT employees using data from project 
tracking software such as Atlassian Jira. Unlike most common representation of employee's worklog as a table of tasks
and hours logged on each of those on specific dates, we define a **physical worklog** which consists of sums of hours 
logged by employee during specific dates physically. Exactly in this worklog representation we want to find a pattern 
of higher productive employees and then spread it to others to maximize general performance.

This physical worklog can be perfectly represented by time series:

![Example of physical worklog](images/physical_worklog.png)

which allows us to extract features from it using common statistical approaches.
We then choose metrics to evaluate employees productivity and build models to be able to restore metric value by 
features.

Key results of this project are set features extracted from time series worklog (see **Feature engineering** below),
regression model (**0.05** average error on scale [0; 1]) and classification model (**0.95** accuracy).

This research was based on data of company N. in 5 quarters (July 2023 - September 2024) about 700+ employees.

Project code is full Python 3.9 and Jupyter Notebook for better visual support.

## Feature engineering

Whole dataset for building models consists of 25 time series based features:

* **Periods**. We analyze so-called "seasonality" in worklog by building a periodogram which allows to extract time 
series periods. A hypothesis is that employees with higher performance log work more often and their worklog periods 
would be less on average. We take 3 most likely periods from periodogram.
* **Stationarity**. Stationary time series are those which maintain average value. In terms of our problem, the more 
employee postpones logging work and accumulates work hours not logged, the less stationary his worklog is. We use three
different stationary tests (Augmented Dickey-Fuller, Phillips-Perron and Kwiatkowski-Phillips-Schmidt-Shin) with 
constant, constant & trend and hyperbolic trend. These fit in 7 features.
* **Structural break**. It helps to understand from peaks in a worklog whether employee went on vacation and logged 
time in advance or it is simply irregular work logging. We use Chow test to find if peak is a structural break.
* **Co-integration**. By comparing the time series and scenarios for logging work with co-integration tests, we can 
understand which scenario it fits more - daily, weekly or monthly. We use Johansen test to compare given time series 
with daily and weekly logging patterns. This fits in 6 features because each pattern result consists of 3 parts: 
    + Whether there is no co-integration relation
    + Whether there is exactly one co-integration relation
    + Whether there is more than one co-integration relation
* **Standard statistics**. We also used simpler statistics: maximum, mean, variance and means for each week work day, 
total of 8 features.

## Project features

* Utilities for data masking and aggregation
* Time series feature engineering
* Target metrics analysis
* Regression models comparison & evaluation
* Feature-based clustering
* Multiclass classification 

## Structure

* [`utils`](utils) for loading, masking and aggregation of data
* [`reports`](reports) for creating physical worklog-based visual reports.
* [`notebooks`](notebooks) for detailed analysis of metrics and models in Jupyter Notebooks.
* [`models`](models) for feature engineering and dataset creation.
* [`metrics`](metrics) for metrics value computation.
* [`data_sample`](data_sample) for example of data used in research (full data can not be shared).

## Requirements

See [`requirements.txt`](requirements.txt).

## References

* T. D. N. Vuong, L. T. Nguyen (2022) The Key Strategies for Measuring Employee Performance in Companies: 
A Systematic Review.
* Box, G. E. P.; Jenkins, G. M.; Reinsel, G. C. (1994). Time Series Analysis: Forecasting and Control (3rd ed.). 
Upper Saddle River, NJ: Prentice–Hall. ISBN 978-0130607744.
* Hayes M. H. Statistical digital signal processing and modeling. — John Wiley & Sons, 2009.
* P. Welch, “The use of the fast Fourier transform for the estimation of power spectra: A method based on time 
averaging over short, modified periodograms”, IEEE Trans. Audio Electroacoust. vol. 15, pp. 70-73, 1967.
* MacKinnon, J.G. 2010. Critical Values for Cointegration Tests. Queen's University, Dept of Economics, 
Working Papers.
* Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity 
against the alternative of a unit root. Journal of Econometrics, 54: 159-178.
* Lütkepohl, H. 2005. New Introduction to Multiple Time Series Analysis. Springer.
* Chow, Gregory C. Tests of Equality Between Sets of Coefficients in Two Linear Regressions — 1960. — Vol. 28. — P. 
591—605.

## License

[MIT License](LICENSE)
