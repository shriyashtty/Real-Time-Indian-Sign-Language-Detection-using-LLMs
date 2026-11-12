import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import base64
import httpx
from tensorflow.keras.models import load_model
import time

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Booting Edge Architecture & Loading Models...")

# 1. Load your 84-coordinate model and class names
model = load_model('isl_coordinate_model.h5')
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# 2. Initialize Modern MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

OLLAMA_URL = "http://localhost:11434/api/generate"
print("Server Ready! Waiting for React Frontend...")

@app.post("/translate")
async def translate_isl_sequence(data: dict):
    """
    Receives an ordered list of raw signs from the frontend, 
    then prompts the local Ollama LLM to synthesize a clean English sentence.
    """
    words_list = data.get("words", [])
    if not words_list:
        return {"translation": ""}
    
    keywords_string = ", ".join(words_list)
    
    # Strict prompt template
    system_prompt = (
        "You are an expert Indian Sign Language interpreter. You will receive a sequence "
        "of individual letters that have been fingerspelled. Your task is to group these "
        "letters into their intended words and arrange those words into a single, grammatically "
        "correct, natural English sentence. Fix any spelling issues caused by missing letters. "
        "Do not include any conversational filler, explanations, or notes. Output ONLY the final sentence.\n"
        f"Fingerspelled Letters: {keywords_string}"
    )
    
    payload = {
        "model": "phi3",
        "prompt": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Forces deterministic, non-random responses
            "num_predict": 50    # Caps output length to maximize processing speed
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                return {"translation": result.get("response", "").strip().replace('"', '')}
            else:
                return {"translation": "Error: Unable to connect to local translation engine."}
        except Exception as e:
            return {"translation": f"Translation offline: {str(e)}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive frame from React
            data = await websocket.receive_text()
            header, encoded = data.split(",", 1)
            file_bytes = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # 2. Process frame through Modern MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            results = detector.detect(mp_image)
            
            if results.hand_landmarks:
                row = []
                # 3. Handle up to 2 hands for 84 total elements
                for hand_landmarks in results.hand_landmarks[:2]:
                    wrist_x = hand_landmarks[0].x
                    wrist_y = hand_landmarks[0].y
                    
                    temp_row = []
                    for lm in hand_landmarks:
                        # Drop Z and anchor to wrist
                        temp_row.extend([lm.x - wrist_x, lm.y - wrist_y])
                        
                    # Scale normalization
                    max_val = max(abs(val) for val in temp_row)
                    if max_val == 0: max_val = 1.0
                    
                    normalized_row = [val / max_val for val in temp_row]
                    row.extend(normalized_row)
                
                # 4. Pad if only one hand is visible
                if len(results.hand_landmarks) == 1:
                    row.extend([0.0] * 42)

                # 5. Predict
                input_data = np.array([row])
                
                start_time = time.perf_counter()
                predictions = model.predict(input_data, verbose=0)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                print(f"Inference Latency: {latency_ms:.2f} ms")
                class_index = np.argmax(predictions)
                confidence = float(predictions[0][class_index])
                detected_sign = class_names[class_index]
                
                await websocket.send_json({
                    "type": "detection",
                    "sign": detected_sign,
                    "confidence": confidence
                })
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Server Error: {e}")