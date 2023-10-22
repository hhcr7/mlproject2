import csv
import json
import requests
import pandas as pd
from requests.api import head
import os
import sys

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact',"train.csv")
    test_data_path: str=os.path.join('artifact',"test.csv")
    raw_data_path: str=os.path.join('artifact',"data.csv")
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        # Getting data from rapidapi
        logging.info("Entered the data ingestion method or component")


        logging.info("Train test split initiated")
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
        logging.info("Ingestion of the data is completed")

        return(
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
        )
