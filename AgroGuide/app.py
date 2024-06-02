from flask import Flask,render_template,request
import numpy as np
import pickle
import sklearn

model = pickle.load(open('model.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
recommend_model = pickle.load(open("recommend_model.pkl",'rb'))
minmaxScalar = pickle.load(open("minmaxscaler.pkl",'rb'))
standardscalar = pickle.load(open("standscaler.pkl",'rb'))
labelencoder = pickle.load(open("label.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/yield_pred")
def predictor():
    return render_template("yield_pred.html")

@app.route("/recommend")
def recommend():
    return render_template("recommend.html")

@app.route("/predict_crop" , methods=['POST'])
def predict_yield():
    if request.method == 'POST':
        Year = request.form['year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['area']
        Item  = request.form['item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = model.predict(transformed_features).reshape(1,-1)

    return render_template('yield_pred.html',prediction = prediction[0][0])

@app.route("/recommend_crop" , methods=["POST"])
def recommend_crop():
    N = request.form['nitrogen']
    P = request.form['phosphorus']
    K = request.form['potassium']
    temp = request.form['temperature']
    humidity = request.form['humidity']
    ph = request.form['ph']
    rainfall = request.form['rainfall']

    features = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    scaled_features = minmaxScalar.transform(features)
    final_features = standardscalar.transform(scaled_features)
    prediction = recommend_model.predict(final_features).reshape(1,-1)

    result = labelencoder.inverse_transform(prediction[0])[0]

    return render_template("recommend.html",prediction = result)


if( __name__ == "__main__"):
    app.run(debug=True)
