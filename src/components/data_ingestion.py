import os 
import sys 
import pandas as pd 
import numpy as np 
from src.logger import logging 
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()

    def IngestionProcess(self):
        try:
            logging.info('reading the data as pandas dataframe')
            df = pd.read_csv(f'C:\Drive D for me\cement price\Data\cleaned.csv')

            logging.info('making the directory')
            os.makedirs(os.path.dirname(self.data_ingestion.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False)


            logging.info('train and test split')

            train_set, test_set = train_test_split(df,test_size=0.20,random_state=2)


            train_set.to_csv(self.data_ingestion.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion.test_data_path,index=False,header=True)

            logging.info('train and test csv filed saved in artifacts folder')

            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )

        except Exception as e:
            logging.info('error occured in ingestion process')
            raise CustomException(e,sys)        