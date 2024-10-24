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

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
