#importing libraries
import numpy as np
import pickle
from flask import Flask,request,jsonify,render_template

#Create flask app
app=Flask(__name__)

#Load the pickle model
model=pickle.load(open('model.pkl','rb'))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output=prediction[0]

    return render_template('index.html',prediction_text="The flower species is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)