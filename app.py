import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import MinMaxScaler

#bikin flask app
app = Flask(__name__)

#panggil model terbaik
model_filename = 'model_sigmoid.pkl'
model = joblib.load(model_filename)
scaler = joblib.load('scaler.pkl')

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict_fuel", methods= ["POST"])
def predict_fuel():
    try:
        # Ambil data dari permintaan untuk melakukan prediksi
        route = float(request.form.get('route', 0))
        vessel_speed = float(request.form.get('vessel_speed', 0))
        wind_speed = float(request.form.get('wind_speed', 0))
        current_speed = float(request.form.get('current_speed', 0))
        swell = float(request.form.get('swell', 0))
        wave = float(request.form.get('wave', 0))

        # Ubah data input menjadi array numpy
        data_input = np.array([route, vessel_speed, wind_speed, current_speed, swell, wave])

        # Lakukan transformasi pada data input menggunakan skaler yang telah Anda buat sebelumnya
        data_input_scaled = scaler.transform(data_input.reshape(1, -1))

        # Lakukan prediksi menggunakan model SVR Anda
        prediction = model.predict(data_input_scaled)

        # Ubah hasil prediksi menjadi bilangan bulat tanpa koma belakang
        prediction = int(prediction[0])

        # Kembalikan hasil prediksi dalam bentuk respons JSON
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Ganti host dan port sesuai kebutuhan Anda
    app.run(host='0.0.0.0', port=5000)