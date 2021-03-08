import pandas as pd 
from flask import Flask,jsonify,request
import joblib

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    req = request.get_json()
    input_data =  req['data']
    input_data_df = pd.DataFrame(input_data, index=[0])

    model = joblib.load('model.pkl')
    scale_obj = joblib.load('scale.pkl')

    input_data_scaled = scale_obj.transform(input_data_df)

    print(input_data_scaled)


    prediction = model.predict(input_data_df)

    if prediction[0] == 1.0:
        Exited_type = 'YES'
    else:
        Exited_type = 'NO'

    return jsonify({'output':{'Exited_type':Exited_type}})

@app.route('/',methods=['GET'])
def home():
    return "Welcome to Bank Churn Prediction"

if __name__== '__main__':
    app.run(host='0.0.0.0', port='9696')
