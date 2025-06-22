import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application



## import ridge regresor model and standard scaler pickle
logistic_model=pickle.load(open('models/logistic.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Id=float(request.form.get('Id','0'))
        SepalLengthCm =float(request.form.get('SepalLengthCm','0'))
        SepalWidthCm =float(request.form.get('SepalWidthCm','0'))
        PetalLengthCm =float(request.form.get('PetalLengthCm','0'))
        PetalWidthCm =float(request.form.get('PetalWidthCm','0'))

        new_data_scaled=standard_scaler.transform([[Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
        result=logistic_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
     app.run(host="0.0.0.0",debug=True)
   
        