import os 
import sys 
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score

def save_object(obj,file_path):
    with open(file_path,'wb') as file:
        pickle.dump(obj,file)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)

            score = r2_score(y_test, y_pred)

            report[model_name] = score


        return report    


    except Exception as e:
        logging.info('error occured in evaluating the model')
        raise CustomException(e,sys)
    


def load_object(file_path):
    with open(file_path,'rb') as file:
        return pickle.load(file)