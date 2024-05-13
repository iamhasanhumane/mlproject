import os
import sys

import numpy as np
import pandas as pd

from src.exception import CustomException 
from sklearn.metrics import r2_score
import dill 


def save_object(file_path , obj):
    try:
        # Getting the directory name
        dir_path = os.path.dirname(file_path) 

        # Making the directory
        os.makedirs(dir_path , exist_ok = True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj , file_obj) 

    except Exception as e:
        raise CustomException(e,sys) 

def evaluate_models(X_train , y_train , X_test , y_test , models):

    try:
        report = {} 

        for i in range(len(list(models))):

            #Get the model object at the index i
            model = list(models.values())[i] 

            # Fit the model in the training dataset
            model.fit(X_train , y_train) 

            # Predict the values in the training dataset
            y_train_pred = model.predict(X_train) 

            # Predict the values in the test dataset
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train , y_train_pred) 

            test_model_score = r2_score(y_test , y_test_pred) 

            report[list(models.keys())[i]] = test_model_score

        return report  

    except Exception as e:
        raise CustomException(e , sys) 



   



 



