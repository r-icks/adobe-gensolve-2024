import React, { useState, useRef } from "react";
import axios from "axios";
import { Stage, Layer, Line } from "react-konva";

const DrawingApp = () => {
  const [lines, setLines] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isLoading, setIsLoading] = useState(false); // Loading state
  const lastPosRef = useRef(null);

  const handleMouseDown = (e) => {
    if (isLoading) return;
    const pos = e.target.getStage().getPointerPosition();
    lastPosRef.current = pos;
    setIsDrawing(true);
    setLines([
      ...lines,
      { points: [{ x: pos.x + 0.0001, y: pos.y + 0.0001 }] },
    ]);
  };

  const handleMouseMove = (e) => {
    if (!isDrawing || isLoading) return;

    const stage = e.target.getStage();
    const pos = stage.getPointerPosition();
    const lastPos = lastPosRef.current;
    const distance = Math.sqrt(
      Math.pow(pos.x - lastPos.x, 2) + Math.pow(pos.y - lastPos.y, 2)
    );

    const newLines = [...lines];
    const lastLine = newLines[newLines.length - 1];

    const x =
      pos.x + (distance > 0 ? ((pos.x - lastPos.x) / distance) * 0.0001 : 0);
    const y =
      pos.y + (distance > 0 ? ((pos.y - lastPos.y) / distance) * 0.0001 : 0);

    lastLine.points = [...lastLine.points, { x, y }];
    newLines.splice(newLines.length - 1, 1, lastLine);
    setLines(newLines);

    lastPosRef.current = pos;
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  const generateCSVData = () => {
    let csvContent = "";

    lines.forEach((line, polylineIndex) => {
      console.log("temp");
      const zero = 0;
      line.points.forEach((point) => {
        const x = point.x.toExponential(17);
        const y = point.y.toExponential(17);
        csvContent += `${polylineIndex},${zero.toExponential(17)},${x / 3},${
          y / 3
        }\n`;
      });
    });

    return csvContent;
  };

  const downloadResults = async () => {
    setIsLoading(true);

    const csvData = generateCSVData();
    const blob = new Blob([csvData], { type: "text/csv" });
    const formData = new FormData();
    formData.append("file", blob, "polylines.csv");

    try {
      const response = await axios.post(
        "http://localhost:5000/upload-csv",
        formData,
        {
          responseType: "blob", // Important to handle binary response
        }
      );

      const zipBlob = new Blob([response.data], { type: "application/zip" });
      const url = window.URL.createObjectURL(zipBlob);
      const a = document.createElement("a");
      a.style.display = "none";
      a.href = url;
      a.download = "results.zip";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error during file download:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <Stage
        width={window.innerWidth}
        height={window.innerHeight}
        onMouseDown={handleMouseDown}
        onMousemove={handleMouseMove}
        onMouseup={handleMouseUp}
        style={{ pointerEvents: isLoading ? "none" : "auto" }}
      >
        <Layer>
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line.points.flatMap((p) => [p.x, p.y])}
              stroke="black"
              strokeWidth={2}
              tension={0.5}
              lineCap="round"
              globalCompositeOperation="source-over"
            />
          ))}
        </Layer>
      </Stage>
      <button
        onClick={downloadResults}
        disabled={isLoading}
        style={{
          position: "fixed",
          bottom: "20px",
          right: "20px",
          padding: "10px 20px",
          backgroundColor: isLoading ? "gray" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          zIndex: 1000,
        }}
      >
        {isLoading ? "Processing..." : "Download Results"}
      </button>
    </div>
  );
};

export default DrawingApp;
