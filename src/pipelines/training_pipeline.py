import os 
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__=='__main__':
    data_ingestion = DataIngestion()
    train_data_file, test_data_file = data_ingestion.IngestionProcess()


    data_transformation = DataTransformation()
    x_train,y_train,x_test,y_test = data_transformation.Transformation_process(train_data_file,test_data_file)

    model_trainer = ModelTrainer()
    model_trainer.Model_Training(x_train,y_train,x_test,y_test)

