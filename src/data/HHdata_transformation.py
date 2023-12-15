import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import BinaryEncoder, MinMaxScaler, FunctionTransformer
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object, rare_replace, dropna_majority, feature_selection

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('data', 'interim', 'proprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, categorical_features, numerical_features):
        try:
            all_features = categorical_features + numerical_features
            DropnaMajority = FunctionTransformer(func = dropna_majority)
            RareReplace = FunctionTransformer(func = rare_replace(categorical_features))

            cat_pipeline = Pipeline(steps = [
                #("dropna_majority", DropnaMajority()),
                ("imputer", SimpleImputer(strategy = "constant", fill_value = "missing")),
                ("rare_replacer", RareReplace()),
                ("binary_encoder", BinaryEncoder()),
                ("scaler", MinMaxScaler())
                ])
            
            num_pipeline = Pipeline(steps = [
                #("dropna_majority", DropnaMajority()),
                ("imputer", SimpleImputer(strategy = "median")),
                #("scaler", MinMaxScaler())
                ])
            
            preprocessor = ColumnTransformer([
                ("dropna_majority", DropnaMajority(), all_features),
                ("cat_pipelines", cat_pipeline, categorical_features),
                ("num_pipeline", num_pipeline, numerical_features),
                ("scaler", MinMaxScaler(), all_features)
                ("feature_selection", SelectFromModel(Lasso(alpha=0.05, random_state=0)), all_features)
                ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            # Convert acres to sqft
            for i in range(len(train_df['lotAreaUnit'])):
                if train_df['lotAreaUnit'][i] == 'acres':
                    train_df['lotAreaValue'][i] = 43560*train_df['lotAreaValue'][i]
                    train_df['lotAreaUnit'][i] = 'sqft'
            for i in range(len(test_df['lotAreaUnit'])):
                if test_df['lotAreaUnit'][i] == 'acres':
                    test_df['lotAreaValue'][i] = 43560*test_df['lotAreaValue'][i]
                    test_df['lotAreaUnit'][i] = 'sqft'

            # Organize into numerical and categorical features
            categorical_features = [feature for feature in train_df.columns if train_df[feature].dtypes == 'O']
            numerical_features = [feature for feature in train_df.columns if train_df[feature].dtypes != 'O']
            miscategorized = ['zipcode', 'longitude', 'latitude']

            categorical_features.extend(miscategorized)
            for feature in miscategorized:
                numerical_features.remove(feature)

            print(numerical_features, categorical_features)
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            irrelevant_features = ['state', 'isFeatured', 'isPremierBuilder', 'isShowcaseListing',
                'isPreforeclosureAuction', 'isNonOwnerOccupied', 'lotAreaUnit',  'homeStatus', 'daysOnZillow', 'country',
                'isUnmappable', 'streetAddress', 'isZillowOwned', 'shouldHighlight', 'zpid', 'homeStatusForHDP',
                'listing_sub_type', 'datePriceChanged', 'currency', 'priceForHDP', 'open_house_info', 'openHouse', 'unit']
            
            train_df.drop(columns = irrelevant_features)
            test_df.drop(columns = irrelevant_features)

            # Drop target variable - the price
            target_feature = 'price'
            
            input_feature_train_df = train_df.drop(columns = [target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]

            input_feature_test_df = test_df.drop(columns = [target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            preprocessing_obj = self.get_data_transformer_object(categorical_features, numerical_features)
            logging.info("Obtained preprocessing object")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Applied preprocessing object on train and test dataframes")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj )
            logging.info(f"Saved preprocessing object")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)