import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
import numpy as np
import os 


class CustomData:
    def __init__(
        self , 
        gender : str , 
        race_ethnicity : str ,   
        parental_level_of_education : str , 
        lunch : str , 
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
    
        self.gender = gender 
        self.race_ethnicity = race_ethnicity 
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score   

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'gender' : [self.gender] ,
                'race/ethnicity' : [self.race_ethnicity] ,  
                'parental level of education': [self.parental_level_of_education] , 
                'lunch' : [self.lunch] , 
                'test preparation course' : [self.test_preparation_course] , 
                'reading score' : [self.reading_score] , 
                'writing score' : [self.writing_score] 
            }

            return pd.DataFrame(custom_data_input_dict) 

        except Exception as e: 
            raise CustomException(e , sys) 


class PredictPipeline:
    def __init__(self): 
        pass

    def predict(self , features):
        try:
            preprocessor_file_path = os.path.join('artifacts' , 'preprocessor.pkl')
            model_file_path = os.path.join("artifacts" , "model.pkl") 
            print(features) 
            print("Before Loading")
            preprocessor = load_object(preprocessor_file_path)
            model = load_object(model_file_path) 
            print("After Loading") 
            data_scaled = preprocessor.transform(features)
            print(data_scaled) 
            predicted_value = model.predict(data_scaled) 

            return predicted_value 

        except Exception as e:
            raise CustomException(e,sys) 
          



    