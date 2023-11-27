import os 
import sys 
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object


class Predicion_pipe():
    def __init__(self):
        pass


    def making_prediction(self,features):
        try:
            logging.info(f'the given features are :{features}')
            df = pd.DataFrame(features,index=[0])
            logging.info(f'the data frame is as follows:{df.head().to_string()}')

            model_path = os.path.join('artifacts','model.pkl')
            model = load_object(model_path)
            result = model.predict(df)


            return result 





        except Exception as e:
            logging.info('error occured in making the production')
            raise CustomException(e,sys)

    