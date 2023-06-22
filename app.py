import tensorflow as tf
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

dic = {0: "Babi", 1: "Sapi"}

model = load_model("inceptionresnetv2.h5")


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "static/image/uploads/" + img.filename
        img.save(img_path)
        p, acc = predict_label(img_path)

    return render_template("index.html", prediction=p, accuracy=acc, img_path=img.filename)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "static/image/uploads/" + img.filename
        img.save(img_path)
        p, acc = predict_label(img_path)
        return {"prediction": p, "accuracy": acc}


def predict_label(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img) / 255.0
    img = img.reshape(1, 300, 300, 3)
    pred = model.predict(img)
    p = tf.argmax(pred, axis=1)
    acc = tf.reduce_max(pred).numpy()

    prediction = dic[p[0].numpy()]
    accuracy = float(acc)

    return prediction, accuracy




if __name__ == "__main__":
    app.run(debug=True)
