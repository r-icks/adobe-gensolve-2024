import React, { useState, useRef } from "react";
import axios from "axios";
import { Stage, Layer, Line } from "react-konva";
import { FaExclamationTriangle, FaPencilAlt } from "react-icons/fa";

const DrawingApp = () => {
  const [lines, setLines] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const lastPosRef = useRef(null);
  const fileInputRef = useRef(null);

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
        `${process.env.REACT_APP_API_URL}/upload-csv`,
        formData,
        {
          responseType: "blob",
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

  const processCSV = async (file) => {
    setIsLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/upload-csv`,
        formData,
        {
          responseType: "blob",
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

  const handleProcessCSVClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      processCSV(file);
    }
  };

  return (
    <div>
      <div
        style={{
          position: "fixed",
          top: "20px",
          left: "50%",
          transform: "translateX(-50%)",
          color: "#007bff",
          fontSize: "24px",
          zIndex: 999,
          display: "flex",
          alignItems: "center",
        }}
      >
        <FaPencilAlt style={{ marginRight: "10px" }} />
        Draw Here
      </div>
      <Stage
        width={window.innerWidth}
        height={window.innerHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
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
      <div
        style={{
          position: "fixed",
          bottom: "70px",
          left: "50%",
          transform: "translateX(-50%)",
          color: "#6c757d",
          fontSize: "14px",
          zIndex: 999,
          display: "flex",
          alignItems: "center",
          pointerEvents: "none",
        }}
      >
        <FaExclamationTriangle style={{ marginRight: "8px" }} />
        It can take up to a minute to hit the backend API for the first time.
      </div>
      <button
        onClick={handleProcessCSVClick}
        disabled={isLoading}
        style={{
          position: "fixed",
          bottom: "20px",
          left: "20px",
          padding: "10px 20px",
          backgroundColor: isLoading ? "gray" : "#28a745",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
          zIndex: 1000,
        }}
      >
        {isLoading ? "Processing..." : "Upload CSV"}
      </button>
      <input
        type="file"
        ref={fileInputRef}
        accept=".csv"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />
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
