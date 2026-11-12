import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const wsRef = useRef(null);

  const [currentSign, setCurrentSign] = useState("...");
  const [confidence, setConfidence] = useState(0);
  const [isConnected, setIsConnected] = useState(false);

  // Translation state management
  const [wordBuffer, setWordBuffer] = useState([]);
  const [finalTranslation, setFinalTranslation] = useState("");
  const [isTranslating, setIsTranslating] = useState(false);

  useEffect(() => {
    wsRef.current = new WebSocket("ws://localhost:8000/ws");

    wsRef.current.onopen = () => setIsConnected(true);
    wsRef.current.onclose = () => setIsConnected(false);

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "detection") {
        setCurrentSign(data.sign);
        setConfidence(data.confidence);
      }
    };

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const captureAndSendFrame = useCallback(() => {
    if (webcamRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        wsRef.current.send(imageSrc);
      }
    }
  }, []);

  useEffect(() => {
    const intervalId = setInterval(captureAndSendFrame, 200);
    return () => clearInterval(intervalId);
  }, [captureAndSendFrame]);

  // Append current detected sign to the sentence buffer
  const addWordToBuffer = () => {
    if (currentSign && currentSign !== "..." && confidence > 0.6) {
      setWordBuffer((prev) => [...prev, currentSign]);
    }
  };

  // Clear all states to start a new sentence
  const clearBuffer = () => {
    setWordBuffer([]);
    setFinalTranslation("");
  };

  // Send collected word sequence to the FastAPI Ollama bridge
  const triggerTranslation = async () => {
    if (wordBuffer.length === 0) return;
    setIsTranslating(true);
    try {
      const response = await fetch("http://localhost:8000/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: wordBuffer }),
      });
      const data = await response.json();
      setFinalTranslation(data.translation);
    } catch (error) {
      setFinalTranslation("Translation failed: Engine connection error.");
    } finally {
      setIsTranslating(false);
    }
  };

  // Capture spacebar key presses to quickly lock in words during demos
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        addWordToBuffer();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [currentSign, confidence]);

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>Real-Time ISL Translator & GenAI Layer</h1>

      <div style={styles.status}>
        System Status: <span style={{ color: isConnected ? "green" : "red" }}>
          {isConnected ? "Connected" : "Disconnected"}
        </span>
      </div>

      <div style={styles.mainLayout}>
        {/* Left Side: Video Capture */}
        <div style={styles.videoContainer}>
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
            style={styles.webcam}
          />
          <div style={styles.controlRow}>
            <button onClick={addWordToBuffer} style={styles.addButton}>
              Add Word [Spacebar]
            </button>
            <button onClick={clearBuffer} style={styles.clearButton}>
              Clear
            </button>
          </div>
        </div>

        {/* Right Side: Translation Workspace */}
        <div style={styles.dashboardContainer}>
          <div style={styles.card}>
            <h3>Live Sign</h3>
            <p style={{ ...styles.signText, color: confidence > 0.6 ? "#007bff" : "#999" }}>
              {currentSign}
            </p>
            <p style={styles.confidenceText}>
              Confidence: {(confidence * 100).toFixed(1)}%
            </p>
          </div>

          <div style={styles.card}>
            <h3>Constructed Sequence</h3>
            <div style={styles.bufferDisplay}>
              {wordBuffer.length > 0 ? wordBuffer.join(" → ") : "Buffer empty. Capture steps above."}
            </div>
            <button
              onClick={triggerTranslation}
              disabled={wordBuffer.length === 0 || isTranslating}
              style={wordBuffer.length === 0 ? styles.disabledButton : styles.translateButton}
            >
              {isTranslating ? "Processing LLM..." : "Generate English Sentence"}
            </button>
          </div>

          {finalTranslation && (
            <div style={styles.translationCard}>
              <h3>Polished Output</h3>
              <p style={styles.translationText}>"{finalTranslation}"</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: { fontFamily: "Arial, sans-serif", padding: "20px", backgroundColor: "#f4f4f9", minHeight: "100vh" },
  header: { textAlign: "center", color: "#333" },
  status: { textAlign: "center", marginBottom: "20px", fontWeight: "bold" },
  mainLayout: { display: "flex", flexDirection: "column", alignItems: "center", gap: "20px", maxWidth: "1200px", margin: "0 auto" },
  videoContainer: { display: "flex", flexDirection: "column", alignItems: "center" },
  webcam: { width: "480px", height: "360px", borderRadius: "10px", border: "4px solid #333", backgroundColor: "#000" },
  controlRow: { display: "flex", gap: "10px", marginTop: "15px", width: "100%" },
  addButton: { flex: 2, padding: "12px", backgroundColor: "#28a745", color: "white", border: "none", borderRadius: "6px", fontWeight: "bold", cursor: "pointer" },
  clearButton: { flex: 1, padding: "12px", backgroundColor: "#dc3545", color: "white", border: "none", borderRadius: "6px", fontWeight: "bold", cursor: "pointer" },
  dashboardContainer: { width: "100%", maxWidth: "480px", display: "flex", flexDirection: "column", gap: "20px" },
  card: { backgroundColor: "white", padding: "20px", borderRadius: "10px", boxShadow: "0 4px 6px rgba(0,0,0,0.05)", borderTop: "4px solid #007bff" },
  translationCard: { backgroundColor: "#e9ecef", padding: "20px", borderRadius: "10px", borderLeft: "6px solid #28a745" },
  signText: { fontSize: "2.5rem", fontWeight: "bold", margin: "10px 0", textAlign: "center" },
  confidenceText: { fontSize: "0.9rem", color: "#666", textAlign: "center" },
  bufferDisplay: { minHeight: "50px", padding: "10px", backgroundColor: "#f8f9fa", border: "1px stroke #ddd", borderRadius: "6px", margin: "10px 0", fontStyle: "italic" },
  translateButton: { width: "100%", padding: "12px", backgroundColor: "#007bff", color: "white", border: "none", borderRadius: "6px", fontWeight: "bold", cursor: "pointer" },
  disabledButton: { width: "100%", padding: "12px", backgroundColor: "#ccc", color: "#666", border: "none", borderRadius: "6px", cursor: "not-allowed" },
  translationText: { fontSize: "1.2rem", fontWeight: "500", color: "#155724" }
};

export default App;