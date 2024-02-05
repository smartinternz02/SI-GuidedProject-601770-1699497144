from flask import Flask, request,render_template
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,StandardScaler

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictSalary')
def predictSalary():
    return render_template('Predict.html')

@app.route('/prdict', methods=['POST'])
def predict():
    model = pickle.load(open("./salary_pred.pickle", 'rb'))
    features = [x for x in request.form.values()]
    np_array = np.array([features])
    print(np_array)


    le_designatiom = LabelEncoder()
    le_designatiom.fit(features)
    le_PG = LabelEncoder()
    le_PG.fit(features)
    le_PostPG = LabelEncoder()
    le_PostPG.fit(features)
    le_location = LabelEncoder()
    le_location.fit(features)

    np_array[:,2] = le_designatiom.transform( np_array[:,2])
    np_array[:,1] = le_location.transform( np_array[:,1])
    np_array[:,3] = le_PG.transform( np_array[:,3])
    np_array[:,4] = le_PostPG.transform( np_array[:,4])

    np_array = np_array.astype(float)

    prediction = model.predict(np_array)

    return render_template('Predict.html', prediction_text=round(prediction[0]))


if __name__ == "__main__":
    print("Starting Flask Server")
    app.run(debug=True)