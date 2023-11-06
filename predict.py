import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_rf.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('income')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict(X)
    

    result = {
        'income_probability': bool(y_pred),
        
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)