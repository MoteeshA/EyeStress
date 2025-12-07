import os
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
import mediapipe as mp
from dotenv import load_dotenv
from flask_mail import Mail, Message
import openai

# ---------------- ENV & OPENAI ----------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # DO NOT hardcode your key

# ---------------- FLASK APP ----------------
app = Flask(__name__)

# Email (Gmail SMTP) config via .env
MAIL_SERVER = 'smtp.gmail.com'
MAIL_PORT = 465
MAIL_USE_TLS = False
MAIL_USE_SSL = True
MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER")

app.config['MAIL_SERVER'] = MAIL_SERVER
app.config['MAIL_PORT'] = MAIL_PORT
app.config['MAIL_USE_TLS'] = MAIL_USE_TLS
app.config['MAIL_USE_SSL'] = MAIL_USE_SSL
app.config['MAIL_USERNAME'] = MAIL_USERNAME
app.config['MAIL_PASSWORD'] = MAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = MAIL_DEFAULT_SENDER

mail = Mail(app)

# =========================
# MediaPipe Face Mesh Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=3,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe 468-landmark topology)
LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373]

# =========================
# Constants (same logic as original)
# =========================
DISPLAY_WIDTH = 720
EMA_ALPHA = 0.35

LOW_FRAC = 0.78
HIGH_FRAC = 0.86
MIN_CLOSED_MS = 50
REFRACTORY_MS = 150

CALIBRATION_SECONDS = 3.0
MATCH_THRESHOLD = 90.0
PERSON_TIMEOUT = 5.0  # seconds without seeing face -> remove person


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(pts):
    A = euclidean(pts[1], pts[5])
    B = euclidean(pts[2], pts[4])
    C = euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


# =========================
# Shared State
# =========================
state = {
    "persons": {},  # person_id -> per-person dict
    "logs": []
}


# no real threading here, but keep the same API
class DummyLock:
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...


frame_lock = DummyLock()
next_person_id = 1


def create_person(now, center):
    return {
        "id": None,  # set by caller
        "center": center,
        "last_seen": now,

        # calibration
        "calib_start": now,
        "calib_buf": deque(maxlen=120),
        "baseline": None,

        # EAR
        "ema_ear": None,

        # blink / stress
        "blinks": 0,
        "stress": "Calibrating...",
        "closed": False,
        "closed_start": 0.0,
        "last_blink_time": 0.0,

        # overlay info (normalized coords)
        "overlay": None,
    }


def process_frame_server(frame_bgr):
    """
    Process a single BGR frame from the browser webcam:
    - detect faces with MediaPipe
    - update global state (persons, EAR, blinks, stress)
    - store normalized overlay coordinates for front-end canvas
    """
    global next_person_id

    # Resize while keeping aspect ratio
    h0, w0 = frame_bgr.shape[:2]
    if w0 != DISPLAY_WIDTH:
        frame_bgr = cv2.resize(
            frame_bgr,
            (DISPLAY_WIDTH, int(frame_bgr.shape[0] * DISPLAY_WIDTH / frame_bgr.shape[1]))
        )
    h, w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    now = time.time()

    with frame_lock:
        res = face_mesh.process(rgb)
        seen_ids = set()

        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                # center of face
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                face_center = (cx, cy)

                # match to existing person by nearest center
                assigned_id = None
                min_dist = float("inf")
                best_pid = None

                for pid, pdata in state["persons"].items():
                    pcx, pcy = pdata["center"]
                    d = euclidean((cx, cy), (pcx, pcy))
                    if d < min_dist:
                        min_dist = d
                        best_pid = pid

                if best_pid is not None and min_dist < MATCH_THRESHOLD:
                    assigned_id = best_pid
                else:
                    assigned_id = next_person_id
                    next_person_id += 1
                    new_person = create_person(now, face_center)
                    new_person["id"] = assigned_id
                    state["persons"][assigned_id] = new_person
                    state["logs"].append(f"[INFO] New person detected -> P{assigned_id}")

                person = state["persons"][assigned_id]
                person["center"] = face_center
                person["last_seen"] = now
                seen_ids.add(assigned_id)

                def get_pts(idx_list):
                    pts = []
                    for i in idx_list:
                        lm = face_landmarks.landmark[i]
                        pts.append((int(lm.x * w), int(lm.y * h)))
                    return pts

                left_pts = get_pts(LEFT_EYE_IDX)
                right_pts = get_pts(RIGHT_EYE_IDX)

                leftEAR = eye_aspect_ratio(left_pts)
                rightEAR = eye_aspect_ratio(right_pts)
                raw_ear = (leftEAR + rightEAR) / 2.0

                # update EMA EAR
                ema_prev = person["ema_ear"]
                ema_ear = raw_ear if ema_prev is None else EMA_ALPHA * raw_ear + (1 - EMA_ALPHA) * ema_prev
                person["ema_ear"] = ema_ear

                # store overlay info (normalized coordinates) for front-end canvas
                left_norm = [[x / w, y / h] for (x, y) in left_pts]
                right_norm = [[x / w, y / h] for (x, y) in right_pts]
                person["overlay"] = {
                    "cx": cx / w,
                    "cy": cy / h,
                    "left_eye": left_norm,
                    "right_eye": right_norm,
                }

                # Calibration
                if person["baseline"] is None:
                    elapsed = now - person["calib_start"]
                    person["calib_buf"].append(ema_ear)

                    if elapsed >= CALIBRATION_SECONDS and len(person["calib_buf"]) > 10:
                        arr = np.array(person["calib_buf"])
                        baseline = float(np.percentile(arr, 95))
                        baseline = float(np.clip(baseline, 0.18, 0.50))
                        person["baseline"] = baseline
                        person["stress"] = "Low / Normal"
                        state["logs"].append(
                            f"[INFO] Baseline EAR set for P{person['id']} -> {baseline:.3f}"
                        )
                else:
                    baseline = person["baseline"]

                    low_thr = baseline * LOW_FRAC
                    high_thr = baseline * HIGH_FRAC
                    if high_thr <= low_thr:
                        high_thr = low_thr + 0.02

                    closed = person["closed"]
                    closed_start = person["closed_start"]
                    last_blink_time = person["last_blink_time"]
                    blinks = person["blinks"]

                    # Blink logic
                    if not closed and raw_ear < low_thr:
                        closed = True
                        closed_start = now
                    elif closed and raw_ear > high_thr:
                        closed_ms = (now - closed_start) * 1000.0
                        gap_ms = (now - last_blink_time) * 1000.0

                        if closed_ms >= MIN_CLOSED_MS and gap_ms >= REFRACTORY_MS:
                            blinks += 1
                            last_blink_time = now
                            state["logs"].append(f"[BLINK] P{person['id']} -> #{blinks}")
                        closed = False

                    person["closed"] = closed
                    person["closed_start"] = closed_start
                    person["last_blink_time"] = last_blink_time
                    person["blinks"] = blinks

                    # Local stress classification (heuristic)
                    if ema_ear < baseline * 0.7:
                        person["stress"] = "Very High / Fatigue"
                    elif ema_ear < baseline * 0.8:
                        person["stress"] = "High"
                    elif ema_ear < baseline * 0.9:
                        person["stress"] = "Moderate"
                    else:
                        person["stress"] = "Low / Normal"

        # Prune persons not seen recently
        now2 = time.time()
        to_delete = []
        for pid, pdata in state["persons"].items():
            if now2 - pdata["last_seen"] > PERSON_TIMEOUT:
                to_delete.append(pid)
        for pid in to_delete:
            del state["persons"][pid]
            state["logs"].append(f"[INFO] P{pid} left frame (removed).")


# =========================
# Helper: snapshot metrics
# =========================
def get_metrics_snapshot():
    with frame_lock:
        persons_payload = []
        for pid, pdata in state["persons"].items():
            persons_payload.append({
                "id": pid,
                "label": f"Person {pid}",
                "ear": round(pdata.get("ema_ear") or 0.0, 3),
                "blinks": int(pdata.get("blinks", 0)),
                "stress": pdata.get("stress", "Calibrating..."),
                "overlay": pdata.get("overlay"),  # normalized coords
            })
        logs_slice = state["logs"][-40:]
    return persons_payload, logs_slice


def generate_ai_stress_summary(persons):
    if not openai.api_key:
        return "AI analysis is not available (missing OpenAI API key)."

    if not persons:
        return (
            "Overall stress rating: Not Measurable\n\n"
            "No face was detected during this session, so stress could not be analyzed. "
            "Try sitting closer to the camera with good lighting and keep your eyes open normally."
        )

    lines = []
    for p in persons:
        lines.append(
            f"{p['label']}: EAR={p['ear']}, blinks={p['blinks']}, stress_tag='{p['stress']}'"
        )
    metrics_text = "\n".join(lines)

    prompt = (
        "You are an assistant interpreting eye blink and EAR-based drowsiness/stress metrics.\n"
        "The model is NOT making any medical or mental-health diagnosis; only give a rough, "
        "everyday reading in terms of alert vs tired.\n\n"
        "Given the following per-person metrics from a short camera session:\n"
        f"{metrics_text}\n\n"
        "1) First, output a single line:\n"
        "   Overall stress rating: Low / Moderate / High / Very High\n"
        "   (choose exactly one word: Low, Moderate, High, or Very High).\n"
        "2) Then, in 2 short paragraphs, explain in simple language what this means.\n"
        "3) Finally, give a bullet list with 3–4 practical tips "
        "(blink more often, take micro-breaks, drink water, adjust lighting, etc.).\n"
        "4) Keep everything friendly and non-clinical.\n"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You provide calm, clear, non-medical wellbeing feedback based on simple metrics."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.6,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        print("OpenAI error:", e)
        return "AI analysis failed due to a technical error."


# =========================
# Flask Routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    """
    Receives a JPEG frame from the browser (via fetch + FormData),
    decodes it, and runs the detection pipeline.
    """
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400

    data = file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "decode failed"}), 400

    process_frame_server(img)
    return jsonify({"status": "ok"})


@app.route("/metrics")
def metrics():
    persons_payload, logs_slice = get_metrics_snapshot()
    return jsonify({
        "persons": persons_payload,
        "logs": logs_slice
    })


@app.route("/send_report", methods=["POST"])
def send_report():
    user_email = request.form.get("email")

    if not user_email:
        return "Email is required", 400

    persons_payload, logs_slice = get_metrics_snapshot()
    ai_summary = generate_ai_stress_summary(persons_payload)

    metrics_lines = ["Session Metrics:"]
    if not persons_payload:
        metrics_lines.append("  No faces detected in the latest snapshot.")
    else:
        for p in persons_payload:
            metrics_lines.append(
                f"  {p['label']}: EAR={p['ear']}, Blinks={p['blinks']}, StressTag={p['stress']}"
            )

    metrics_text = "\n".join(metrics_lines)
    logs_text = "\n".join(logs_slice) if logs_slice else "No recent log events."

    body = f"""
Hello,

Here is your latest eye blink & stress detection report.

{metrics_text}

-------------------------
AI Interpretation (non-medical):
-------------------------
{ai_summary}

-------------------------
Recent Events:
-------------------------
{logs_text}

This report was generated automatically from your webcam session.
This is NOT a medical diagnosis, only an approximate indication of eye fatigue / alertness.

Regards,
Multi-Person Eye Blink & Stress Detection System
"""

    try:
        msg = Message(
            subject="Your Eye Blink & Stress Detection Report",
            recipients=[user_email],
        )
        msg.body = body
        mail.send(msg)
        return render_template("index.html", email_status="Report sent successfully!")
    except Exception as e:
        print("Email send error:", e)
        return render_template("index.html", email_status=f"Failed to send email: {e}")


if __name__ == "__main__":
    # Local run
    app.run(debug=True, host="0.0.0.0", port=5011)
