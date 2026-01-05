# ===== Fix PosixPath issue on Windows =====
import pathlib
class FakePosixPath(type(pathlib.Path())):
    def __new__(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
pathlib.PosixPath = FakePosixPath

# ===== Imports =====
import torch 
import cv2
import numpy as np
import time
import math
import socket
import pickle
import struct
from twilio.rest import Client

# ===== Audio Alert (Windows WAV) =====
try:
    import winsound
    def beep():
        try:
            winsound.PlaySound('sound.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"[ERROR] Could not play sound: {e}")
except ImportError:
    print("[WARNING] winsound module not available. Audio alert disabled.")
    def beep():
        pass

# ===== Configuration =====
CONF_THRESHOLD = 0.5
BOTTOM_RATIO = 0.70                 # bottom 70% considered
STAMPEDE_THRESHOLD = 0.85           # 85% occupancy
SPEED_THRESHOLD = 120.0             # px/sec considered abnormal
MAX_ASSOC_DIST = 100.0              # px, tracker association distance
MAX_MISSED = 10                     # frames before a lost track is removed

# ===== User-defined capacity =====
while True:
    try:
        MAX_CAPACITY = int(input("Enter the maximum capacity for the area: "))
        if MAX_CAPACITY <= 0:
            print("Capacity must be a positive integer.")
            continue
        break
    except ValueError:
        print("Invalid input. Enter a number.")

# ===== Twilio Setup =====
TWILIO_SID = "AC0d96fcd86a117b0c9884e9f0ea89eb1e"
TWILIO_AUTH_TOKEN = "0efc9e782175a5d404a17270c28235b6"
TWILIO_FROM = "+13252413360"
TWILIO_TO = "+917483889815"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_alert(msg):
    """Send SMS via Twilio"""
    try:
        message = client.messages.create(
            body=msg,
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        print(f"[INFO] SMS sent: {message.sid}")
    except Exception as e:
        print(f"[ERROR] SMS failed: {e}")

# ===== Simple centroid-based tracker =====
class SimpleTracker:
    def __init__(self, max_assoc_dist=100.0, max_missed=10):
        self.next_id = 0
        self.tracks = {}
        self.max_assoc_dist = max_assoc_dist
        self.max_missed = max_missed

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, boxes, timestamp):
        det_centers = [self._center(b) for b in boxes]
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[i]['center'] for i in track_ids]
        assigned_tracks = {}
        assigned_dets = set()

        if track_centers and det_centers:
            dists = np.zeros((len(track_centers), len(det_centers)), dtype=np.float32)
            for i, c1 in enumerate(track_centers):
                for j, c2 in enumerate(det_centers):
                    dists[i, j] = math.hypot(c1[0]-c2[0], c1[1]-c2[1])

            used_tracks = set()
            used_dets = set()
            while True:
                i_min, j_min = np.unravel_index(np.argmin(dists), dists.shape)
                min_val = dists[i_min, j_min]
                if not np.isfinite(min_val) or min_val > self.max_assoc_dist:
                    break
                if i_min in used_tracks or j_min in used_dets:
                    dists[i_min, j_min] = np.inf
                    continue
                tid = track_ids[i_min]
                assigned_tracks[tid] = j_min
                used_tracks.add(i_min)
                used_dets.add(j_min)
                dists[i_min, :] = np.inf
                dists[:, j_min] = np.inf
            assigned_dets = used_dets

        for tid, det_idx in assigned_tracks.items():
            box = boxes[det_idx]
            center = det_centers[det_idx]
            prev_center = self.tracks[tid]['center']
            prev_time = self.tracks[tid]['last_time']
            dt = max(1e-6, timestamp - prev_time)
            speed = math.hypot(center[0]-prev_center[0], center[1]-prev_center[1]) / dt
            self.tracks[tid].update({
                'bbox': box,
                'center': center,
                'last_time': timestamp,
                'missed': 0,
                'last_speed': speed
            })

        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]['missed'] += 1

        to_remove = [tid for tid, t in self.tracks.items() if t['missed'] > self.max_missed]
        for tid in to_remove:
            del self.tracks[tid]

        for j, box in enumerate(boxes):
            if j not in assigned_dets:
                center = det_centers[j]
                self.tracks[self.next_id] = {
                    'bbox': box,
                    'center': center,
                    'last_time': timestamp,
                    'missed': 0,
                    'last_speed': 0.0
                }
                self.next_id += 1

        return {tid: {
                    'bbox': t['bbox'],
                    'center': t['center'],
                    'speed': t['last_speed'],
                    'missed': t['missed']
                } for tid, t in self.tracks.items()}

# ===== Load YOLOv5 =====
print("[INFO] Loading YOLOv5 crowd detection model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = CONF_THRESHOLD

# ===== Raspberry Pi Camera Feed Setup =====
HOST = '10.92.156.139'  # Pi IP
PORT = 9992
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

data = b""
payload_size = struct.calcsize("L")

tracker = SimpleTracker(max_assoc_dist=MAX_ASSOC_DIST, max_missed=MAX_MISSED)
print("[INFO] Ready. Press ESC to exit.")

last_stampede_state = False

while True:
    # ===== Receive frame from Pi =====
    while len(data) < payload_size:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet

    if not data:
        continue

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data)

    ts = time.time()
    h, w = frame.shape[:2]
    bottom_start = int(h * (1.0 - BOTTOM_RATIO))

    results = model(frame)
    df = results.pandas().xyxy[0]

    people_boxes = []
    for _, row in df.iterrows():
        if str(row['name']).lower() == "person":
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if cy >= bottom_start:
                box = (x1, y1, x2, y2)
                people_boxes.append(box)

    person_count = len(people_boxes)
    occupancy_ratio = person_count / MAX_CAPACITY
    tracks = tracker.update(people_boxes, ts)

    # ===== Stampede Detection =====
    stampede = occupancy_ratio >= STAMPEDE_THRESHOLD
    abnormal_ids = []
    if stampede:
        for tid, t in tracks.items():
            if t['missed'] == 0 and t['speed'] >= SPEED_THRESHOLD:
                abnormal_ids.append((tid, t['speed']))

    # ===== Capacity Exceeded Alert =====
    if person_count > MAX_CAPACITY:
        alert_msg = f"⚠️ Capacity Exceeded! People={person_count}, Max={MAX_CAPACITY}"
        print(alert_msg)
        send_alert(alert_msg)
        beep()

    # ===== Alerts =====
    if stampede and not last_stampede_state:
        alert_msg = f"🚨 STAMPEDE ALERT! People={person_count}, Capacity={MAX_CAPACITY}, Occ={occupancy_ratio:.2f}"
        print(alert_msg)
        send_alert(alert_msg)
        beep()
    elif not stampede and last_stampede_state:
        print("[INFO] Stampede cleared.")

    last_stampede_state = stampede

    if abnormal_ids:
        for tid, spd in abnormal_ids:
            alert_msg = f"⚠️ Abnormal Activity: ID={tid}, Speed={spd:.1f}px/s"
            print(alert_msg)
            send_alert(alert_msg)
        beep()

    # ===== Drawing (for visualization) =====
    box_color = (0, 0, 255) if stampede else (0, 255, 0)
    for tid, t in tracks.items():
        if t['missed'] > 0:
            continue
        x1, y1, x2, y2 = t['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        label = f"ID {tid}"
        if stampede:
            label += f" | {t['speed']:.0f}px/s"
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    status = f"People {person_count} / Cap {MAX_CAPACITY}  Occ {occupancy_ratio:.2f}"
    status_color = (0, 0, 255) if stampede else (0, 200, 0)
    cv2.putText(frame, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    y0 = 60
    for tid, spd in abnormal_ids[:6]:
        cv2.putText(frame, f"ABN ID {tid}: {spd:.0f}px/s",
                    (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y0 += 26

    cv2.imshow("Crowd Stampede + Tracking + Abnormal Activity", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

client_socket.close()
cv2.destroyAllWindows()
