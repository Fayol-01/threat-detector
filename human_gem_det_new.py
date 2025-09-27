import os
import cv2
import time
import json
import queue
import threading
import requests
import keyboard
from dotenv import load_dotenv
from pydantic import BaseModel
from ultralytics import YOLO
from google import genai
from google.genai import types

# ----------------- Gemini & Discord -------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

client = genai.Client(api_key=GEMINI_KEY)
WEBHOOK_URL = "https://discord.com/api/webhooks/1391033437690794024/SyIEJR0bA5cBRZAiLB8t9f3xyXwRMlH1xV41-udSHe30bgWeud_OtwsS2zp9rFgcDIaZ"  # <--- Replace

class Output(BaseModel):
    action: str
    threat_rating: int
    description: str

# ----------------- Worker Thread ----------------------
clip_queue = queue.Queue()

def gemini_worker():
    while True:
        clip_path = clip_queue.get()
        if clip_path is None:   # poison pill for shutdown
            break
        print(f"[Worker] Sending {clip_path} to Gemini…")
        try:
            with open(clip_path, "rb") as f:
                video_bytes = f.read()

            response = client.models.generate_content(
                model="models/gemini-2.5-pro",
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(data=video_bytes,
                                                   mime_type="video/mp4"),
                            video_metadata=types.VideoMetadata(),
                        ),
                        types.Part(
                            text=("This is a CCTV clip from my front door. "
                                  "Explain what is happening, rate the threat out of 10, "
                                  "and describe any threat.")
                        ),
                    ]
                ),
                config={
                    "response_mime_type": "application/json",
                    "response_schema": Output,
                },
            )

            try:
                data = json.loads(response.text)
            except Exception as e:
                print("[Worker] JSON parse error:", e, response.text)
                os.remove(clip_path)
                continue

            with open(clip_path, "rb") as f:
                files = {"file": (os.path.basename(clip_path), f, "video/mp4")}
                content = (
                    f"@everyone Intruder Alert!\n{data['action']}"
                    if data["threat_rating"] > 5
                    else f"Someone at the door!\n{data['action']}"
                )
                r = requests.post(WEBHOOK_URL, data={"content": content}, files=files)
                if r.ok:
                    print(f"[Worker] Sent {clip_path} to Discord.")
                else:
                    print("[Worker] Discord error:", r.status_code, r.text)

        finally:
            # remove local file
            if os.path.exists(clip_path):
                os.remove(clip_path)
            clip_queue.task_done()

worker = threading.Thread(target=gemini_worker, daemon=True)
worker.start()

# ----------------- Main Thread ------------------------
model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 10

print("Main thread running. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame, verbose=False)
    person_detected = any(cls == 0 for cls in results[0].boxes.cls.int().tolist())

    if person_detected:
        print("Person detected! Recording 10-second clip…")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        clip_name = f"clip_{int(time.time())}.mp4"
        out = cv2.VideoWriter(clip_name, fourcc, fps,
                              (frame.shape[1], frame.shape[0]))

        start = time.time()
        while time.time() - start < 10:
            ret2, frame2 = cap.read()
            if not ret2:
                break
            out.write(frame2)
            cv2.imshow("YOLO Human Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        out.release()

        clip_queue.put(clip_name)  # hand off to worker
        print(f"Queued {clip_name} for Gemini processing.")

    else:
        cv2.imshow("YOLO Human Detection", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord("q") or keyboard.is_pressed("esc"):
        print("Exit key pressed.")
        break

cap.release()
cv2.destroyAllWindows()

# Shutdown worker
clip_queue.put(None)
worker.join()
print("Shutdown complete.")
