import cv2 as cv
import numpy as np
import pandas as pd
import svgwrite
from svgpathtools import svg2paths
import os

def image_to_svg(img, contours_to_draw, circle_info, bounding_box, linesToDraw, filename="output.svg"):
    try:
        height, width = img.shape[:2]
        dwg = svgwrite.Drawing(filename, profile='full', size=(width, height))
        for contour, color in contours_to_draw:
            points = contour[:, 0, :].tolist()
            path_data = f"M {points[0][0]},{points[0][1]} " + " ".join([f"L {p[0]},{p[1]}" for p in points[1:]])
            path_data += " Z"
            path = dwg.path(d=path_data, stroke=svgwrite.rgb(*color, '%'), fill="none", stroke_width=1)
            dwg.add(path)
        for center, radius in circle_info:
            dwg.add(dwg.circle(center=center, r=radius, stroke=svgwrite.rgb(0, 0, 255, '%'), fill="none", stroke_width=1))
        for box in bounding_box:
            points = box.tolist()
            for i in range(4):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % 4]
                dwg.add(dwg.line((x1, y1), (x2, y2), stroke=svgwrite.rgb(255, 0, 0, '%'), stroke_width=1))
        for line in linesToDraw:
            dwg.add(dwg.line((int(a) for a in line[0]), (int(a) for a in line[1]), stroke=svgwrite.rgb(0, 255, 0, '%'), stroke_width=1))
        dwg.save()
    except Exception as e:
        raise RuntimeError(f"Failed to generate SVG file: {str(e)}")

def svg2polylines(svg_path):
    try:
        paths, attributes = svg2paths(svg_path)
        polylines = []
        for path in paths:
            polyline = []
            for segment in path:
                start_point = segment.start
                end_point = segment.end
                polyline.append((start_point.real, start_point.imag))
                if segment.__class__.__name__ != 'Line':
                    for t in np.linspace(0, 1, num=100):
                        point = segment.point(t)
                        polyline.append((point.real, point.imag))
                polyline.append((end_point.real, end_point.imag))
            polylines.append(np.array(polyline))
        return polylines
    except Exception as e:
        raise RuntimeError(f"Failed to convert SVG to polylines: {str(e)}")

def convert_arrays_to_csv(arrays, output_csv_path):
    try:
        data = []
        for idx, arr in enumerate(arrays):
            for point in arr:
                data.append([idx, 0, float(point[0]), float(point[1])])
        df = pd.DataFrame(data)
        df.to_csv(output_csv_path, index=False, header=None)
    except Exception as e:
        raise RuntimeError(f"Failed to save arrays to CSV: {str(e)}")

def main():
    path = input("Enter path to CSV file: ").strip()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")
    
    try:
        polylines = pd.read_csv(path, header=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {str(e)}")

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
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    shape_info = []
    
    for contour in contours:
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
            circularity = 4 * np.pi * area / (peri ** 2)
            if 0.43 < circularity < 0.79:
                shape_info.append(("unidentified", contour))
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
    finalContours = []
    linesToDraw = []

    for shape, contour in shape_info:
        if shape == "triangle":
            cv.drawContours(mask, [contour[0]], -1, 0, 1)
            contoursToDraw.append((contour[1], (0, 128, 0)))
        elif shape == "rectangle":
            rect = cv.minAreaRect(contour[0])
            box = cv.boxPoints(rect)
            box = box.astype(int)
            boundingBox.append(box)
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
            finalContours.append((contour, (255, 255, 0)))

    img = cv.bitwise_and(img, img, mask=mask)

    for info in circleInfo:
        center, radius = info
        cv.circle(img, center, radius - 5, (0, 0, 255), 1)

        cv.line(img, (center[0] - radius, center[1]), (center[0] + radius, center[1]), (0, 255, 0), 1)
        cv.line(img, (center[0], center[1] - radius), (center[0], center[1] + radius), (0, 255, 0), 1)
        linesToDraw.append([(int(center[0] - radius), int(center[1])), (int(center[0] + radius), int(center[1]))])
        linesToDraw.append([(int(center[0]), int(center[1] - radius)), (int(center[0]), int(center[1] + radius))])

    for box in boundingBox:
        cv.drawContours(img, [box], 0, (255, 0, 0), 1)

        p1_h = tuple(box[1])
        p2_h = tuple(box[3])
        cv.line(img, p1_h, p2_h, (0, 255, 0), 1)
        linesToDraw.append([p1_h, p2_h])
        mid1 = tuple(((box[0] + box[1]) // 2).astype(int))
        mid2 = tuple(((box[1] + box[2]) // 2).astype(int))
        mid3 = tuple(((box[2] + box[3]) // 2).astype(int))
        mid4 = tuple(((box[3] + box[0]) // 2).astype(int))

        cv.line(img, mid1, mid3, (0, 255, 0), 1) 
        cv.line(img, mid2, mid4, (0, 255, 0), 1)
        linesToDraw.append([mid1, mid3])
        linesToDraw.append([mid2, mid4])

    for contour in contoursToDraw:
        cv.drawContours(img, [contour[0]], -1, contour[1], 1)
        finalContours.append(contour)

        M = cv.moments(contour[0])
        if M['m00'] != 0:  
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            contour_points = np.squeeze(contour[0]).astype(np.float32)
            mean, eigenvectors = cv.PCACompute(contour_points, mean=np.array([]).astype(np.float32))
            principal_axis = eigenvectors[0]
            length = 100  
            x1 = int(cx - length * principal_axis[0])
            y1 = int(cy - length * principal_axis[1])
            x2 = int(cx + length * principal_axis[0])
            y2 = int(cy + length * principal_axis[1])
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            linesToDraw.append([(x1, y1), (x2, y2)])

    image_to_svg(img, finalContours, circleInfo, boundingBox, linesToDraw)

    try:
        polylines = svg2polylines("output.svg")
        convert_arrays_to_csv(polylines, output_csv_path="output.csv")
        os.remove('output.svg')
    except Exception as e:
        raise RuntimeError(f"Failed to process SVG file: {str(e)}")

if __name__ == "__main__":
    try:
        main()
        print("Script executed successfully!")
    except Exception as e:
        print(f"Script execution failed: {str(e)}")
