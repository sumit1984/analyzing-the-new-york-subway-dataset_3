import numpy as np
import pandas


"""
This function tries to predict the ridership in subway using a linear model and gradient descent:
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

def compute_cost(features, values, theta):
    """
    Compute the cost function given a set of features / values, 
    and the values for our thetas.
    """
    
    # your code here
    error = (values - features.dot(theta))
    cost    = error.dot(error)   
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """
    
    m = len(values)
    cost_history = []

    for i in range(num_iterations):
        # your code here
        cost = compute_cost(features, values, theta)/(2.0*m)
        cost_history.append([cost])
        
        error = features.dot(theta) - values
        error = np.reshape(error,(error.shape[0], 1))
        errorWeighted = features*error
        errorSum      = (np.sum(errorWeighted,0))/(m*1.0)
        theta = theta - alpha*errorSum        
        
    return theta, pandas.Series(cost_history)

def compute_r_squared(data, prediction):
    '''
    This functon calculates the R^2 value, to estimate the goodness of prediction
    '''
    
    # your code here
    meanData = np.mean(data)
    ssTotal  = np.sum((data-meanData).dot(data-meanData))
    
    ssres    = np.sum((data - prediction).dot(data-prediction))
    
    r_squared = 1 - (ssres)/(1.0*ssTotal)
    return r_squared

def predictions(dataframe):

    '''
    following function initializes the features and the response variable
    '''
    temp = pandas.DatetimeIndex(dataframe['DATEn'])
    dataframe['weekday'] = temp.weekday
    dataframe['hour2'] = dataframe[['Hour']].apply(lambda x: x*x)
    dataframe['hour3'] = dataframe[['Hour']].apply(lambda x: x*x*x)
    dataframe['wd2']   = dataframe[['weekday']].apply(lambda x: (x*x))
    
    features = dataframe[['rain','meantempi','Hour', 'hour2', 'hour3', 'weekday', 'wd2' ]]
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
    

    # Set values for alpha, number of iterations.
    alpha = 0.1 # please feel free to change this value
    num_iterations = 100 # please feel free to change this value

    # Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, 
                                                            values_array, 
                                                            theta_gradient_descent, 
                                                            alpha, 
                                                            num_iterations)

    
    prediction = np.dot(features_array, theta_gradient_descent)
    r2 = compute_r_squared(values, prediction)
    return prediction, plot, theta_gradient_descent, r2


filename = 'turnstile_data_master_with_weather.csv'
weather_data = pandas.read_csv(filename)

[result, plot, theta, r2] = predictions(weather_data)

print 'The r^2 value is : ', r2