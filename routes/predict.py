# app/routes/predict_routes.py
from flask import Blueprint, request, jsonify
import base64
from PIL import Image
from io import BytesIO


from ultralytics import YOLO

predict_bp = Blueprint("predict", __name__)
model = YOLO('best.pt')


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
        formatted_detected_objects = [{"wound_name": key} for key, value in detected_objects.items()]

        return jsonify({"detected_objects": formatted_detected_objects})

    return jsonify({"error": "No image file provided"}), 400