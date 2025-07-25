import numpy as np 
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso , ElasticNet
from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass 
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_training(self,train_arr , test_arr):
        try:
            logging.info('Splitting Dependent and Independent vairable from train and test array')
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],           
            )

            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'DecisionTreeRegressor':DecisionTreeRegressor()
            }

            models_report:dict = evaluate_model(X_train,y_train ,X_test, y_test,models)
            print('model_report')
            print('\n==============================================================\n')

            # To get best model score from dictionary
            best_model_score = max(sorted(models_report.values()))

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name :{best_model_name} , R2 Score :{best_model_score} ')
            print('\n==============================================================')
            logging.info(f'Best Model Found , Model Name :{best_model_name} , Model score : {best_model_score}')

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
        except Exception as e:
                logging.info('Error occured in ModelTrainer')
                raise CustomException(e,sys)
