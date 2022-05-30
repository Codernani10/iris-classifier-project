from copyreg import pickle
from flask import Flask, render_template,request
from sklearn.ensemble import RandomForestClassifier
import pickle

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict" ,methods=["GET" ,'POST'])
def predict():
    if request.method =="POST":
        sepal_length = request.form["spl"]
        sepal_width = request.form["spw"]
        petal_length = request.form["ptl"]
        petal_width = request.form["ptw"]
        
        data = [[float(sepal_length),float(sepal_width),float(petal_length),float(petal_width)]]
        rf =pickle.load(open("pklfile.pkl","rb"))
        prediction = rf.predict(data)[0]

        return render_template("prediction.html",prediction=prediction)

if __name__ == "__main__" :
    app.run(debug=True)