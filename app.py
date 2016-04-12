from flask import Flask, render_template, request,\
    current_app, Response
from recognizer import Recognizer
import json


app = Flask(__name__)
with app.app_context():
    current_app.recognizer = Recognizer(
            '/images',
            '/app/haarcascade_frontalface_default.xml'
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_user', methods=['POST'])
def check_user():
    img_buffer = request.files['webcam']
    res = {}
    faces = app.recognizer.detect(img_buffer)
    res['faces'] = faces
    return Response(json.dumps(res),  mimetype='application/json')

if __name__ == '__main__':
    app.run(
            host='0.0.0.0',
            debug=True,
            ssl_context=('/certs/server.crt', '/certs/server.key')
    )
