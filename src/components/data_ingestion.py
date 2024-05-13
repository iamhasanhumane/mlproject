import os
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the Data Ingestion Method or Component')

        try:
            # Reading the data from our local folder
            df = pd.read_csv('notebook\data\stud.csv')    
            logging.info('Read the Dataset as a DataFrame ')   

            # Creating the artifacts folder/directory to store the data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)

            # storing the data frame df in raw_data.csv in artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path , index = False , header = True)

            logging.info('Train Test Split Initiated')
            train_set , test_set = train_test_split(df , test_size = 0.2 , random_state = 42)

            #  storing the train_set in train.csv in artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path , index = False , header = True)

            #  storing the test_set in test.csv in artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info('Ingestion of the data is completed')

            return(
                self.ingestion_config.train_data_path , 
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e , sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array , test_array , _ = data_transformation.initiate_data_transformation(train_data , test_data)    

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array , test_array)) 





