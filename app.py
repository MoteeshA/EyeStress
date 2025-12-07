import cv2
import time
import threading
from collections import deque

import numpy as np
from flask import Flask, Response, jsonify, render_template
import mediapipe as mp

app = Flask(__name__)

# =========================
# MediaPipe Face Mesh Setup
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=3,  # allow up to 3 faces
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices (MediaPipe 468-landmark topology)
LEFT_EYE_IDX = [33, 159, 145, 133, 153, 144]
RIGHT_EYE_IDX = [362, 386, 374, 263, 380, 373]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(pts):
    """
    pts: list of 6 (x,y)
    EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    """
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

current_frame = None
running = True
frame_lock = threading.Lock()
next_person_id = 1


def create_person(now, center):
    """Initialize a new person state."""
    return {
        "id": None,  # will be set by caller
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
    }


# =========================
# Camera Processing Loop
# =========================
def camera_loop():
    global current_frame, running, next_person_id

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        running = False
        return

    DISPLAY_WIDTH = 720
    EMA_ALPHA = 0.35

    # Thresholds (per person)
    # Raw EAR is used for blink detection, EMA only for smoothing display.
    LOW_FRAC = 0.78        # close threshold
    HIGH_FRAC = 0.86       # reopen threshold
    MIN_CLOSED_MS = 50     # minimum closed duration to count blink
    REFRACTORY_MS = 150    # minimum gap between blinks

    CALIBRATION_SECONDS = 3.0
    MATCH_THRESHOLD = 90.0  # pixels, to match faces to existing persons

    print("[INFO] Camera loop started")

    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.resize(
            frame,
            (DISPLAY_WIDTH, int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1]))
        )
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        now = time.time()

        seen_ids = set()

        if res.multi_face_landmarks:
            for face_landmarks in res.multi_face_landmarks:
                # Compute center of the face
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                face_center = (cx, cy)

                # Assign to nearest existing person or create new
                assigned_id = None
                min_dist = float("inf")
                best_pid = None

                with frame_lock:
                    for pid, pdata in state["persons"].items():
                        pcx, pcy = pdata["center"]
                        d = euclidean((cx, cy), (pcx, pcy))
                        if d < min_dist:
                            min_dist = d
                            best_pid = pid

                if best_pid is not None and min_dist < MATCH_THRESHOLD:
                    assigned_id = best_pid
                else:
                    # Create new person
                    with frame_lock:
                        assigned_id = next_person_id
                        next_person_id += 1
                        new_person = create_person(now, face_center)
                        new_person["id"] = assigned_id
                        state["persons"][assigned_id] = new_person
                        state["logs"].append(f"[INFO] New person detected -> P{assigned_id}")

                with frame_lock:
                    person = state["persons"].get(assigned_id)
                    if person is None:
                        person = create_person(now, face_center)
                        person["id"] = assigned_id
                        state["persons"][assigned_id] = person

                    person["center"] = face_center
                    person["last_seen"] = now

                seen_ids.add(assigned_id)

                # ---- Eye-specific processing ----
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
                raw_ear = (leftEAR + rightEAR) / 2.0  # raw EAR for blink logic

                with frame_lock:
                    # Smooth EAR for display/stress only
                    ema_prev = person["ema_ear"]
                    ema_ear = raw_ear if ema_prev is None else EMA_ALPHA * raw_ear + (1 - EMA_ALPHA) * ema_prev
                    person["ema_ear"] = ema_ear

                    # Draw eye contours
                    cv2.polylines(frame, [np.array(left_pts, dtype=np.int32)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_pts, dtype=np.int32)], True, (0, 255, 0), 1)

                    # ----- Calibration for this person -----
                    if person["baseline"] is None:
                        elapsed = now - person["calib_start"]
                        person["calib_buf"].append(ema_ear)

                        cv2.putText(frame, f"P{person['id']} Calibrating...",
                                    (cx - 80, cy - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        if elapsed >= CALIBRATION_SECONDS and len(person["calib_buf"]) > 10:
                            arr = np.array(person["calib_buf"])
                            baseline = float(np.percentile(arr, 95))
                            # Slightly higher clamp for open eyes
                            baseline = float(np.clip(baseline, 0.18, 0.50))
                            person["baseline"] = baseline
                            person["stress"] = "Low / Normal"
                            state["logs"].append(
                                f"[INFO] Baseline EAR set for P{person['id']} -> {baseline:.3f}"
                            )
                    else:
                        baseline = person["baseline"]

                        # thresholds based on baseline (using RAW EAR for transitions)
                        low_thr = baseline * LOW_FRAC
                        high_thr = baseline * HIGH_FRAC
                        if high_thr <= low_thr:
                            high_thr = low_thr + 0.02

                        closed = person["closed"]
                        closed_start = person["closed_start"]
                        last_blink_time = person["last_blink_time"]
                        blinks = person["blinks"]

                        # Blink detection: use raw_ear vs thresholds
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

                        # Store updated blink state
                        person["closed"] = closed
                        person["closed_start"] = closed_start
                        person["last_blink_time"] = last_blink_time
                        person["blinks"] = blinks

                        # Simple stress classification (using EMA)
                        if ema_ear < baseline * 0.7:
                            person["stress"] = "Very High / Fatigue"
                        elif ema_ear < baseline * 0.8:
                            person["stress"] = "High"
                        elif ema_ear < baseline * 0.9:
                            person["stress"] = "Moderate"
                        else:
                            person["stress"] = "Low / Normal"

                        # HUD overlays for this person near the face
                        label = f"P{person['id']}"
                        cv2.putText(frame, label,
                                    (cx - 40, cy - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(frame, f"Blinks: {person['blinks']}",
                                    (cx - 80, cy - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(frame, f"EAR: {ema_ear:.3f}",
                                    (cx - 80, cy + 0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"Stress: {person['stress']}",
                                    (cx - 80, cy + 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No face detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Prune persons not seen for some time
        now2 = time.time()
        with frame_lock:
            to_delete = []
            for pid, pdata in state["persons"].items():
                if now2 - pdata["last_seen"] > 5.0:  # 5 seconds timeout
                    to_delete.append(pid)
            for pid in to_delete:
                del state["persons"][pid]
                state["logs"].append(f"[INFO] P{pid} left frame (removed).")

        # Encode frame as JPEG and store in shared state
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        with frame_lock:
            current_frame = buffer.tobytes()

        time.sleep(0.01)

    cap.release()
    print("[INFO] Camera loop stopped")


# Start the camera thread immediately
camera_thread = threading.Thread(target=camera_loop, daemon=True)
camera_thread.start()


# =========================
# Flask Routes
# =========================

@app.route("/")
def index():
    return render_template("index.html")


def gen_frames():
    global current_frame
    while True:
        with frame_lock:
            frame = current_frame
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.02)
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/metrics")
def metrics():
    with frame_lock:
        persons_payload = []
        for pid, pdata in state["persons"].items():
            persons_payload.append({
                "id": pid,
                "label": f"Person {pid}",
                "ear": round(pdata.get("ema_ear") or 0.0, 3),
                "blinks": int(pdata.get("blinks", 0)),
                "stress": pdata.get("stress", "Calibrating...")
            })
        logs_slice = state["logs"][-40:]

    return jsonify({
        "persons": persons_payload,
        "logs": logs_slice
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5011)
