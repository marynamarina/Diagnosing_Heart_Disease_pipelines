import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
std_scaler, reduce_dim, model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    X_test_scaled = std_scaler.transform(final_features)
    X_test_reduced = reduce_dim.transform(X_test_scaled)
    
    prediction = model.predict(X_test_reduced)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Diagnosing Heart Disease: yes(1), no(0) {}'.format(output))

if __name__ == "__main__":
    app.run()
