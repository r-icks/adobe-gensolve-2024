from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import cv2 as cv
from io import BytesIO
import zipfile

load_dotenv()

app = Flask(__name__)

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS(app, resources={r"/*": {"origins": frontend_url}})


def process_csv_and_generate_image(polylines):
    """
    Process the CSV file to generate an image.
    Returns the image as a binary stream.
    """
    img = np.zeros((512, 512), dtype=np.uint8)
    current_polyline = None

    for i in range(len(polylines)):
        if [polylines.iloc[i, 0], polylines.iloc[i, 1]] != current_polyline:
            current_polyline = [polylines.iloc[i, 0], polylines.iloc[i, 1]]
        else:
            pt1 = (int(round(polylines.iloc[i-1, 2])), int(round(polylines.iloc[i-1, 3])))
            pt2 = (int(round(polylines.iloc[i, 2])), int(round(polylines.iloc[i, 3])))
            cv.line(img, pt1, pt2, color=255, thickness=1)

    blur = cv.blur(img, (1, 1))
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    shape_info = []

    for i, contour in enumerate(contours):
        if cv.contourArea(contour) < 3:
            shape_info.append(("unidentified", contour))
            continue

        eps = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, eps, True)

        shape = "unidentified"
        peri = cv.arcLength(contour, True)
        area = cv.contourArea(contour)
        vertices = len(approx)

        if vertices >= 11:
            (x, y), radius = cv.minEnclosingCircle(contour)
            circle_area = np.pi * (radius ** 2)
            if abs(area - circle_area) < 0.2 * circle_area:
                center = (int(x), int(y))
                shape = "circle"
                shape_info.append((shape, (center, int(radius), contour)))
                continue
        else:
            eps = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, eps, True)
            shape = "unidentified"
            peri = cv.arcLength(contour, True)
            area = cv.contourArea(contour)
            vertices = len(approx)
            if vertices == 3:
                shape = "triangle"
            elif vertices == 4:
                shape = "rectangle"
            elif vertices == 5:
                shape = "pentagon"
            elif vertices == 6:
                shape = "hexagon"
            elif vertices == 7:
                if peri / area > 0.05:
                    shape_info.append(("unidentified", contour))
                    continue
                shape = "heptagon"
            elif vertices == 8:
                shape = "octagon"
            elif vertices == 9:
                shape = "nonagon"
            elif vertices == 10:
                if peri / area > 0.105:
                    shape_info.append(("unidentified", contour))
                    continue
                shape = "decagon"

        if shape != "unidentified":
            shape_info.append((shape, (contour, approx)))
        else:
            shape_info.append((shape, contour))

    mask = np.ones((512, 512), dtype=np.uint8) * 255
    circleInfo = []
    boundingBox = []
    contoursToDraw = []

    for shape, contour in shape_info:
        if shape == "triangle":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (0, 128, 0)))
        elif shape == "rectangle":
            xi, yi, wi, hi = cv.boundingRect(contour[0])
            boundingBox.append((xi, yi, wi, hi))
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
        elif shape == "pentagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (128, 0, 128)))
        elif shape == "hexagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (0, 128, 128)))
        elif shape == "heptagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (255, 165, 0)))
        elif shape == "octagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (0, 165, 255)))
        elif shape == "nonagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (75, 0, 130)))
        elif shape == "decagon":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (102, 102, 102)))
        elif shape == "circle":
            center, radius = contour[0], contour[1]
            circleInfo.append((center, radius))
            cv.drawContours(mask, [contour[2]], -1, 0, 1)
        else:
            cv.drawContours(img, [contour], -1, (255, 255, 0), 1)

    img = cv.bitwise_and(img, img, mask=mask)

    for info in circleInfo:
        cv.circle(img, info[0], info[1] - 5, (0, 0, 255), 1) # red
    for box in boundingBox:
        cv.rectangle(img, (box[0] + 2, box[1] + 2), (box[0] + box[2] - 2, box[1] + box[3] - 2), (255, 0, 0), 1) # blue
    for contour in contoursToDraw:
        cv.drawContours(img, [contour[0]], -1, contour[1], 1)

    output_image_path = os.path.join('backend', 'output_image.jpg')
    cv.imwrite(output_image_path, img)

    _, img_encoded = cv.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()

    return img_bytes


@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({"error": "File is not a CSV"}), 400

    try:
        # Read the CSV file using pandas
        polylines = pd.read_csv(file, header=None)

        # Process the CSV to generate the image
        img_bytes = process_csv_and_generate_image(polylines)

        # Prepare an empty CSV (this is a placeholder; you can generate your CSV dynamically)
        csv_content = "col1,col2\n".encode('utf-8')

        # Create a BytesIO object to hold the ZIP file in memory
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add the image to the ZIP
            zf.writestr('output_image.jpg', img_bytes)

            # Add the CSV to the ZIP
            zf.writestr('output.csv', csv_content)

        # Ensure the buffer is at the beginning for reading
        zip_buffer.seek(0)

        # Send the ZIP file as a response
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='output.zip'
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


port = os.getenv("PORT", 5000)
if __name__ == '__main__':
    app.run(debug=True, port=port)