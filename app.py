from flask import Flask, render_template, url_for, request, redirect
import joblib
import pandas as pd
import pycaret
from pycaret.classification import load_model
import numpy as np

app = Flask(__name__)

model = joblib.load('tuned_dt_wildfire.pkl')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/understanding')
def understanding():
    return render_template('understanding.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/prediction', methods=['GET', 'POST'] )
def prediction():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    elif request.method == "POST":
        try:
            # Get form data
            data = request.form.to_dict()
            
            # Create a dataframe from form data
            df = pd.DataFrame([data])

            # Execute prediction
            prediction = model.predict(df)

            # Get class data based on prediction
            classData = getClassData(prediction[0])

            # probabilities = model.predict_proba(df)
            # confidence = max(probabilities[0]) * 100
            
            return render_template('prediction.html', 
                                     prediction=classData,
                                     form_data=data)

        except Exception as e:
            return render_template('prediction.html',
                                    error=str(e))

def getClassData(prediction):
    if not prediction:
        raise ValueError("Invalid prediction!")
    
    fire_classes = {
        "A": {
            "fireClass": "A",
            "rangeHa": "0.01 - 0.1 ha",
            "severity": "Incipient Fire"
        },
        "B": {
            "fireClass": "B", 
            "rangeHa": "0.11 - 4.0 ha",
            "severity": "Small Fire"
        },
        "C": {
            "fireClass": "C",
            "rangeHa": "4.1 - 40.0 ha", 
            "severity": "Moderate Fire"
        },
        "D": {
            "fireClass": "D",
            "rangeHa": "40.1 - 200.0 ha",
            "severity": "Large Fire"
        },
        "E": {
            "fireClass": "E",
            "rangeHa": "200.1+ ha",
            "severity": "Extreme Fire"
        }
    }

    if prediction not in fire_classes:
        raise ValueError(f"Invalid prediction class: '{prediction}'")
    
    return fire_classes[prediction]

if __name__ == '__main__':
    app.run(debug=True)