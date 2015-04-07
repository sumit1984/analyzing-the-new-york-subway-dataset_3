import numpy as np
import pandas as pd
import statsmodels.api as sm
from ggplot import *

"""
In this question, you need to:
1) implement the compute_cost() and gradient_descent() procedures
2) Select features (in the predictions procedure) and make predictions.

"""

def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma
    
def compute_r_squared(data, predictions):
    '''
    
    This functon calculates the R^2 value, to estimate the goodness of prediction
    
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.
    
    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''
    
    # your code here
    meanData = np.mean(data)
    ssTotal  = np.sum((data-meanData).dot(data-meanData))
    
    ssres    = np.sum((data - predictions).dot(data-predictions))
    
    r_squared = 1 - (ssres)/(1.0*ssTotal)
    return r_squared   

def predictions(dataframe):
 
    # Select Features (try different features!)
    temp = pandas.DatetimeIndex(dataframe['DATEn'])
    dataframe['weekday'] = temp.weekday
    dataframe['hour2'] = dataframe[['Hour']].apply(lambda x: x*x)
    dataframe['hour3'] = dataframe[['Hour']].apply(lambda x: x*x*x)
    dataframe['wd2']   = dataframe[['weekday']].apply(lambda x: (x*x))
    features = dataframe[['rain', 'precipi', 'Hour', 'hour2', 'hour3', 'meantempi', 'weekday', 'wd2' ]]
    #features = dataframe[['rain']]
    
    # Add UNIT to features using dummy variables
    dummy_units = pd.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
    
    model = sm.OLS(values_array,features_array)
    results = model.fit()
    
    prediction = np.dot(features_array, results.params)
    
    r2 = compute_r_squared(values, prediction)

    return prediction, r2

filename = 'turnstile_data_master_with_weather.csv'
weather_data = pd.read_csv(filename)

prediction, r2 = predictions(weather_data)
print 'The r^2 value is : ', r2




