# add logs
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)

def dropna_majority(df):
    cols_to_drop = []
    try:
        for i in range(len(df.columns)):
            if df[df.columns[i]].isna().sum() > len(df)//2:
                cols_to_drop.append(df.columns[i])
        # Add a log here
        df.drop(columns = cols_to_drop, inplace = True)
        return df
    
    except Exception as e:
        raise CustomException(e, sys)

def rare_replace(df, categorical_features):
    try:
        for feature in categorical_features:
            # For each categorical feature, we count the no. of prices associated to it
            # Divide the count by the total length to get the % of the dataset that belongs to this category
            temp = df.groupby(feature)['price'].count()/len(df)

            # temp_df consists of cat features which are over 1% of the dataset
            temp_df = temp[temp>0.01].index

            # If the feature is in temp_df, leave it intact, else replace it with Rare_var
            df[feature] = np.where(df[feature].isin(temp_df), df[feature], 'Rare_var')

            return df
    
    except Exception as e:
        raise CustomException(e, sys)