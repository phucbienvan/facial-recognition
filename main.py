from flask import Flask
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import base64
from flask import request

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADER'] = 'Content-type'

def base64_to_image(img_base64):
    try:
        img_base64 = np.fromstring(base64.b64decode(img_base64), dtype=np.uint8)
        img_base64 = cv2.imdecode(img_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return img_base64

def count_face(face):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 10)
    return len(faces) 

@app.route('/face', methods=["POST"])
@cross_origin(origins="*")
def face_recognition():
    img_base64 = request.form.get('img')

    face = base64_to_image(img_base64)

    count = count_face(face)
    print(count)

    return "count face: " + str(count)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port='9999', debug=True)
