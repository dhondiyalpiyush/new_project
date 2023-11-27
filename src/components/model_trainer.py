import os 
import sys 
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from src.utils import save_object
from src.utils import evaluate_model
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()


    def Model_Training(self,x_train,y_train,x_test,y_test):
        try:
            models = {'Decision tree': DecisionTreeRegressor(),
                      'Random Forest': RandomForestRegressor()}
            
            report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)

            best_score = max(sorted(report.values()))

            best_model_name = max(sorted(report))

            logging.info(f'model accuracy: {report}')
            logging.info(f'best model found {best_model_name}, model score = {best_score}')
            
            best_model = models[best_model_name]

            print(f'best model found = {best_model}')
            print('\n===========================================================\n')
            print(f'best model score = {best_score}')

            save_object(best_model,self.model_trainer.trained_model_file_path)



        except Exception as e:
            logging.info('error occured in Model Training')
            raise CustomException(e,sys)
         
