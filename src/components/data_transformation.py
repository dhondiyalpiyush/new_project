import os 
import sys 
import pandas as pd 
import numpy as np 
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass





@dataclass
class DataTransformationConfig:
    pass


class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()


    def Transformation_process(self,train_data_file_path,test_data_file_path):
        try:
            train_df = pd.read_csv(train_data_file_path)
            test_df = pd.read_csv(test_data_file_path)

            x_train,y_train,x_test,y_test = (
                train_df.drop(columns='Concrete compressive strength',axis=1),
                train_df['Concrete compressive strength'],
                test_df.drop(columns='Concrete compressive strength',axis=1),
                test_df['Concrete compressive strength']
            )
            logging.info('x_train,y_train,x_test,y_test csv received')
            logging.info(f'x_train_head :{x_train.head().to_string()}')
            logging.info(f'x_test_head:{x_test.head().to_string()}')

            return(
                x_train,y_train,x_test,y_test
            )


        except Exception as e:
            logging.info('error occured in data transformation process')
            raise CustomException(e,sys)
