
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 17:40:39 2017

@author: cdavid
"""

""" Import all packages and the used settings and functions """

import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')


from settings import Settings
from src.functions.data_preprocessing import *
from src.functions.functions_plot import *
from src.functions.classification_pipeline import *

settings = Settings()


""" ---------------------------------------------------------------------------------------------------------------
Load training and test dataset
"""

# Load train and test dataset
df_train = pd.read_csv(settings.config['Data Locations'].get('train'))
df_validate = pd.read_csv(settings.config['Data Locations'].get('test'))

df_train.name = 'df_train'
df_validate.name = 'df_validate'


# Target variable: Category

""" ---------------------------------------------------------------------------------------------------------------
Explore the data
First take a look at the training dataset
- what are the features and how many features does the training data include
- are the missing values (but take a deeper look at the data preperation process)
- what are the different units of the features
"""

# Get a report of the training and test dataset as csv
# -> Use the function describe_report(df, name, output_file_path=None)
describe_report(df_train, output_file_path=settings.csv)
describe_report(df_validate, output_file_path=settings.csv)

# Show if there are different columns in the training and test dataset. If there is only one difference, it is likely, that its the target variable.
# If there are columns in the test dataset, which are not in the training dataset, they have to be deleted, because the algorithm will not see them during the training.
# -> Use the function column_diff(df_train, df_test)
column_diff(df_train, df_validate)

# Create boxplots to indentify outliers. Histograms are a good standard way to see if feature is skewed but to find outliers, boxplots are the way to use
# -> Use the function create_boxplots(df, output_file_path=None)
#create_boxplots(df_train, output_file_path=settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Feature Creation
"""

def create_datetimes(df):
    df['Dates'] = pd.to_datetime(df.Dates)
    df['day'] = df['Dates'].dt.day
    df['month'] = df['Dates'].dt.month
    df['hour'] = df['Dates'].dt.hour
    df['minute'] = df['Dates'].dt.minute
    return df

df_train = create_datetimes(df_train)
df_validate = create_datetimes(df_validate)



df_train.drop(['Dates', 'Descript', 'Resolution'], axis=1, inplace=True)
df_validate.drop(['Dates', 'Id'], axis=1, inplace=True)


df_train = labelEnc(df_train)
df_validate = labelEnc(df_validate)

df_train = df_train[:1000]
df_validate = df_validate[:1000]

""" ---------------------------------------------------------------------------------------------------------------
Machine Learning (Classification)
"""

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df_train, test_size=0.2, random_state=42)

y_train = train_set['Category']
x_train = train_set.drop(['Category'], axis=1)

y_test = test_set['Category']
x_test = test_set.drop(['Category'], axis=1)

x_validate = df_validate

pipeline = Pipeline([
    ('reduce_dim', PCA()),
    ('feature_scaling', MinMaxScaler()), # scaling because linear models are sensitive to the scale of input features
    ('classification', KNeighborsClassifier()),
    ])

param_grid = [{'reduce_dim__n_components': [5, 9],
               'classification__n_neighbors': [20, 30, 50],
               'classification__leaf_size': [5, 20, 30],
               'classification__p': [1, 2]
              }]


pipe_best_params = classification_pipeline(x_train, y_train, pipeline, 5, 'accuracy', param_grid)

pipe_best = Pipeline([
    ('reduce_dim', PCA(n_components = pipe_best_params['reduce_dim__n_components'])),
    ('feature_scaling', MinMaxScaler()),
    ('classification', KNeighborsClassifier(
        n_neighbors = pipe_best_params['classification__n_neighbors'],
        leaf_size = pipe_best_params['classification__leaf_size'],
        p = pipe_best_params['classification__p'],))
])

print(pipe_best_params['reduce_dim__n_components'])
print(pipe_best_params['classification__n_neighbors'])
print(pipe_best_params['classification__leaf_size'])
print(pipe_best_params['classification__p'])

train_errors = evaluate_pipe_best_train(x_train, y_train, pipe_best, 'KNeighborsClassifier', binary=False)


plot_learning_curve(pipe_best, 'KNeighborsClassifier', x_train, y_train, 'accuracy', output_file_path=settings.figures)



""" ---------------------------------------------------------------------------------------------------------------
Evaluate the System on the Test Set
"""
#Evaluate the model with the test_set
# -> Use the function evaluate_pipe_best_test(x_train, y_train, pipe_best, algo, output_file_path=None)
evaluate_pipe_best_test(x_test, y_test, pipe_best)


##### Use other classifier