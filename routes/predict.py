# app/routes/predict_routes.py
from flask import Blueprint, request, jsonify
import base64
from PIL import Image
from io import BytesIO


from ultralytics import YOLO

predict_bp = Blueprint("predict", __name__)
model = YOLO('best_model.pt')# Load the model


@predict_bp.route("/predict", methods=["POST"])
def predict():
    if "image" in request.files:
        # Get the image file from the request
        image_file = request.files["image"]

        # Read the image and perform inference
        img = Image.open(image_file)
        output = model.predict(img, conf=0.3, imgsz=720)
        names = model.names


        # Process the output as needed
        detected_objects = {}

        for r in output:
            for c in r.boxes.cls:
                class_name = names[int(c)]
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

        # Convert the detected_objects dictionary to the desired format
        if detected_objects:
            most_frequent_key = max(detected_objects, key=detected_objects.get)
            # Format hasil deteksi sesuai dengan format yang diinginkan
            return jsonify({"detected_objects": most_frequent_key})
        else:
            return jsonify({"detected_objects": "tidak ada objek terdeteksi"})

    return jsonify({"error": "No image file provided"}), 400

        # Format hasil deteksi sesuai dengan format yang diinginkan
