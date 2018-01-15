# Deep Time-to-Failure
Predicting survival functions and remaining time-to-failure using statistical techniques and Recurrent Neural Networks in Python.

The tutorial is divided into:
1. Fitting survival distributions and regression survival models using lifelines.
2. Predicting the distribution of future time-to-failure using raw time-series of covariates as input of a Recurrent Neural Network in keras.

The second part is an extension of the [wtte-rnn](https://github.com/ragulpr/wtte-rnn) framework developed by @ragulpr.
The original work focused on time-to-event models for churn predictions while we will focus on the time-to-failure variant.

In a time-to-failure model the single sequence will always end with the failure event while in a time-to-event model each sequence will contain multiple target events and the goal is to estimating when the next event will happen.
This small simplification allows us to train a RNN of arbitrary lengths to predict only a fixed event in time.

The tutorial is a also a re-adaptation of the work done by @daynebatten on predicting run to failure time of jet engines.

The approach can be used to predict failures of any component in many other application domains or, in general, to predict any time to an event that determines the end of the sequence of observations. Thus, any model predicting a single target time event.

# References

Lifelines documentation: http://lifelines.readthedocs.io/en/latest/index.html

Data source original: https://c3.nasa.gov/dashlink/resources/139/

Data source tutorial: https://github.com/daynebatten/keras-wtte-rnn/blob/master/data_readme.txt

[wtte-rnn](https://github.com/ragulpr/wtte-rnn)

[keras-wtte-rnn](https://github.com/daynebatten/keras-wtte-rnn)

# Installation

Required dependencies are: keras, tensorflow, matplotlib, seaborn, scikit-learn, pandas, numpy, wtte, lifelines.
The notebook examples were developed in Python 3.5.

# Theory

## Survival analysis

The techniques described in this tutorial are based on the Servival analysis theory and applications. Survival analysis is the study of expected duration of time until one or more events happen, in our case failure in mechanical systems.

Survival analysis differs from traditional regression problems because the data can be partially observed, or censored. That is, we observe a phenomenon up to a point in time where we stop collecting data either because the failure event happens and is registered (uncensored) or because it was not possible to collect data furtherly (censored) due to technical issues or because of the failure is not happened yet at current time. We are not considering the case where data is left-censored. We assume we can always observe the beginning of the signal or events sequence.

The survival model is then characterized by either its \b{survival function S(t)} defined as the probability that a subject survives longer than time t; or by its \b{cumulative hazard function H(t)} which is the accumulation of hazard over time and defined by the integral between 0 and t of the probability of failing instantly at time x given that it survived up to x.

From the survival function we can derive the distribution of future lifetime, which is the time remaining until death given survival up to current time.

In literature we often find two main non-parametric estimators for fitting survival distributions such as the Kaplan-Maier and Nelson-Aaler.

The Weibull distribution is used instead for modeling a parametric distribution that well represent the future life time in many real-word use cases.
The Weibull distribution is defined as by parameters alpha and beta as such:

![img](http://latex.codecogs.com/gif.latex?f%28t%29%3D%5Cfrac%7B%5Cbeta%7D%7B%5Calpha%7D%20%5Cleft%20%28%5Cfrac%7Bt%7D%7B%5Calpha%7D%20%5Cright%29%5E%7B%5Cbeta%20-%201%7D%20e%20%5E%7B-%5Cleft%20%28%5Cfrac%7Bt%7D%7B%5Calpha%7D%20%5Cright%29%5E%7B%5Cbeta%7D%7D) for his continous form, or

![img](http://latex.codecogs.com/gif.latex?p%28t%29%3D%20e%20%5E%7B-%5Cleft%20%28%5Cfrac%7Bt%7D%7B%5Calpha%7D%20%5Cright%29%5E%7B%5Cbeta%7D%7D%20-%20e%20%5E%7B-%5Cleft%20%28%5Cfrac%7Bt%20&plus;%201%7D%7B%5Calpha%7D%20%5Cright%29%5E%7B%5Cbeta%7D%7D) for the discrete form.

The parameter alpha is a multipler of where the expected value and mode of the distribution is positioned in time while the beta parameter is an indicator of the variance or confidence of our predictions.

## Survival regression

Fitting a distribution can be worth in case that we want to compare different populations, e.g. in clinic tests, and decide whether there is a statistical signficance between the different survival curves. In order to predict the time-to-failure we want to use different features as regressors and predict as output the survival function, or related functions, of each individual.

For static attributes or hand-crafted features (e.g. lags, accumulated statistics, aggregated statistics) we can use the [Cox's Proportional Hazard model](https://en.wikipedia.org/wiki/Survival_analysis#Cox_proportional_hazards_(PH)_regression_analysis) or [Aalen's Additive model] (http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#aalen-s-additive-model).

Both Cox's and Aalen's models are based on a survival function non-parametric baseline which is multiplied by another function which is a combination of the input features. Both of them do a good job on describing the variables that impact the survival of a given individual. Nevertheless, they are limited in type of input data can handle and suffer from low generalization and high computational cost due to the high degree of freedom due to the non-parametric nature of the problem.

DeepTTF consists in using the raw time-series of the covariates and static attributes as input features and predicting as output the parameters alpha and beta that characterize the Weibull distribution of the future time-to-failure (TTF).

You can read more about the theory and intuitions behind the Weibul time-to-event RNN at https://github.com/ragulpr/wtte-rnn or in his presentation at the ML Jeju Camp 2017: https://docs.google.com/presentation/d/1H_TK9eQCMGTcslc4AnMCNTUskWIYcJAxsV18ac-fIqM/edit#slide=id.g1fa2ecfbc0_0_38.


# Related work

DeepSurv: https://www.researchgate.net/publication/303812000_Deep_Survival_A_Deep_Cox_Proportional_Hazards_Network
