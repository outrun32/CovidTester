from flask import Flask, render_template, request, flash, url_for, redirect

from flask_uploads import UploadSet, configure_uploads, IMAGES

from flask_ngrok import run_with_ngrok

import numpy as np

from PIL import Image

import os

import cv2

from predicter import Predicter

ALLOWED_EXTENSIONS = set([".png", ".jpg", ".jpeg"])
UPLOAD_FOLDER = "files_to_predict"


app = Flask(__name__)
run_with_ngrok(app)

photos = UploadSet('photos', IMAGES)
app.config["IMAGE_UPLOADS"] = UPLOAD_FOLDER

TEMPLATE_FILE = "index.html"

predicter = Predicter()


@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == "POST" and request.files:
        image = request.files["image"]
        img = Image.open(image)
        img = np.array(img)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        return str(predicter.predict_image(img))
    return render_template(TEMPLATE_FILE)

app.run()

