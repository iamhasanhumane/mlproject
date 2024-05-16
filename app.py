from flask import Flask , render_template , request 
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import StandardScaler 
from src.pipeline.predict_pipeline import CustomData , PredictPipeline 


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict' , methods = ["POST","GET"])
def predict_datapoint(): 
    if request.method == "GET":
        return render_template('home.html')  
    else:
        data = CustomData(
            gender = request.form['gender'] , 
            race_ethnicity = request.form['ethnicity'] , 
            parental_level_of_education = request.form["parental_level_of_education"], 
            lunch = request.form['lunch'] , 
            test_preparation_course = request.form['test_preparation_course'],
            reading_score = request.form['reading_score'],
            writing_score = request.form['writing_score']
        )
        pred_df = data.get_data_as_dataframe() 
        print(pred_df)
        print("Before Prediction")


        predict_pipeline = PredictPipeline()
        print("Mid Prediction")

        result = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=result[0]) 




if __name__ == "__main__":
    app.run(debug = True)     