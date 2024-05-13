import os 
import sys

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor , 
    GradientBoostingRegressor , 
    RandomForestRegressor, 
) 

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object , evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path : str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    # we get the train_array and test_array from data_transformation.
    def initiate_model_trainer(self , train_array , test_array):
        try:
            logging.info('Splitting the train and test array from target column')
            X_train , y_train , X_test , y_test = (
                train_array[:,:-1] , 
                train_array[:,-1] , 
                test_array[:,:-1] , 
                test_array[:,-1] ,
            )

            models = {
                "Random Forest" : RandomForestRegressor() , 
                "Decision Tree" : DecisionTreeRegressor() ,
                "Gradient Boosting" : GradientBoostingRegressor() , 
                "Linear Regression" : LinearRegression() ,
                "K-Neighbors Regressor" : KNeighborsRegressor() ,
                "XGB Regressor" : XGBRegressor() ,
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report : dict = evaluate_models(X_train = X_train , y_train = y_train ,
                                                  X_test = X_test , y_test = y_test ,
                                                  models = models) 

            # Getting the best model score from the dictionary
            best_model_score = max(list(model_report.values()))   

            # Getting the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ] 
            best_model = models[best_model_name] 

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and test dataset")   

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path , 
                obj = best_model
            ) 

            predicted = best_model.predict(X_test) 

            r2_square = r2_score(y_test , predicted) 

            return r2_square 


        except Exception as e:
            raise CustomException(e , sys)  






