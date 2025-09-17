import base64, re, numpy as np, cv2
from camera import Predictor
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
predictor = Predictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    try:
        data = request.json["image"]
        # Remove base64 header
        img_str = re.search(r'base64,(.*)', data).group(1)
        nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run prediction
        pred = predictor.predict_frame(frame)

        return jsonify({"prediction": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
