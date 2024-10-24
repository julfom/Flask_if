from flask import Flask, render_template, request
from pickle import load
import numpy as np

app = Flask(__name__)

model = load(open("models/linear_regression_mall.sav", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    sex = int(request.form['sex'])
    age = int(request.form['age'])
    income = float(request.form['income'])
    features = np.array([[sex, age, income]])
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("models/linear_regression_mall.sav", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)>50:
            prediction='High score'
        else:
            prediction='Low score'
            
        return render_template("result.html",prediction=prediction)