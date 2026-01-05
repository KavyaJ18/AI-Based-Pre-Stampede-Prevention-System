from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
import sqlite3, os, time, math, pathlib, socket, pickle, struct
import cv2, torch, numpy as np
import winsound
from werkzeug.security import generate_password_hash, check_password_hash
from twilio.rest import Client

# ===== Global Variables =====
ALERT_FLAG = False  # Flag for frontend buzzer
last_stampede_state = False

# ===== Fix PosixPath for Windows =====
class FakePosixPath(type(pathlib.Path())):
    def __new__(cls, *args, **kwargs):
        return pathlib.WindowsPath(*args, **kwargs)
pathlib.PosixPath = FakePosixPath

# ===== Flask Config =====
app = Flask(__name__)
app.secret_key = "supersecretkey"
DB_PATH = "users.db"

# ===== Ensure DB exists =====
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
init_db()

# ===== Audio Alert =====
def beep():
    try:
        winsound.PlaySound('static/sound.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)
    except:
        pass

# ===== Twilio Setup =====
TWILIO_SID = "AC0d96fcd86a117b0c9884e9f0ea89eb1e"
TWILIO_AUTH_TOKEN = "0efc9e782175a5d404a17270c28235b6"
TWILIO_FROM = "+13252413360"
TWILIO_TO = "+917483889815"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_alert(msg):
    global ALERT_FLAG
    try:
        client.messages.create(body=msg, from_=TWILIO_FROM, to=TWILIO_TO)
        print(f"[INFO] SMS sent: {msg}")
        ALERT_FLAG = True
    except Exception as e:
        print(f"[ERROR] SMS failed: {e}")
        ALERT_FLAG = False

# ===== YOLOv5 & Tracker Config =====
CONF_THRESHOLD = 0.5
BOTTOM_RATIO = 0.7
STAMPEDE_THRESHOLD = 0.85
SPEED_THRESHOLD = 120.0
MAX_ASSOC_DIST = 100.0
MAX_MISSED = 10
MAX_CAPACITY = 20

class SimpleTracker:
    def __init__(self, max_assoc_dist=100.0, max_missed=10):
        self.next_id = 0
        self.tracks = {}
        self.max_assoc_dist = max_assoc_dist
        self.max_missed = max_missed

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2)//2, (y1 + y2)//2)

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
                    dists[i,j] = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
            used_tracks, used_dets = set(), set()
            while True:
                i_min,j_min=np.unravel_index(np.argmin(dists),dists.shape)
                if not np.isfinite(dists[i_min,j_min]) or dists[i_min,j_min]>self.max_assoc_dist: break
                if i_min in used_tracks or j_min in used_dets:
                    dists[i_min,j_min]=np.inf
                    continue
                tid = track_ids[i_min]
                assigned_tracks[tid]=j_min
                used_tracks.add(i_min)
                used_dets.add(j_min)
                dists[i_min,:]=np.inf
                dists[:,j_min]=np.inf
            assigned_dets = used_dets

        for tid, det_idx in assigned_tracks.items():
            box = boxes[det_idx]; center = det_centers[det_idx]
            prev_center = self.tracks[tid]['center']
            prev_time = self.tracks[tid]['last_time']
            dt = max(1e-6, timestamp-prev_time)
            speed = math.hypot(center[0]-prev_center[0], center[1]-prev_center[1])/dt
            self.tracks[tid].update({'bbox':box,'center':center,'last_time':timestamp,'missed':0,'last_speed':speed})

        for tid in track_ids:
            if tid not in assigned_tracks: self.tracks[tid]['missed']+=1

        for tid, t in list(self.tracks.items()):
            if t['missed']>self.max_missed: del self.tracks[tid]

        for j, box in enumerate(boxes):
            if j not in assigned_dets:
                self.tracks[self.next_id]={'bbox':box,'center':det_centers[j],'last_time':timestamp,'missed':0,'last_speed':0.0}
                self.next_id+=1

        return {tid:{'bbox':t['bbox'],'center':t['center'],'speed':t['last_speed'],'missed':t['missed']} for tid,t in self.tracks.items()}

tracker = SimpleTracker(MAX_ASSOC_DIST, MAX_MISSED)
print("[INFO] Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)
model.conf = CONF_THRESHOLD

# ===== Raspberry Pi Feed Config =====
PI_HOST = '10.92.156.139'
PI_PORT = 9992

 
def gen_frames(max_capacity):
    """Generate frames from Raspberry Pi feed with detection and alerts"""
    global last_stampede_state, ALERT_FLAG
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((PI_HOST, PI_PORT))
    data = b""
    payload_size = struct.calcsize("L")

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet
        if not data: continue

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
                cy = (y1 + y2)//2
                if cy >= bottom_start:
                    people_boxes.append((x1,y1,x2,y2))

        person_count = len(people_boxes)
        occupancy_ratio = person_count/max_capacity
        tracks = tracker.update(people_boxes, ts)
        stampede = occupancy_ratio >= STAMPEDE_THRESHOLD
        abnormal_ids = [(tid, t['speed']) for tid,t in tracks.items() if t['missed']==0 and t['speed']>=SPEED_THRESHOLD] if stampede else []

        # Alerts
        if person_count>max_capacity:
            send_alert(f"⚠️ Capacity Exceeded! People={person_count}, Max={max_capacity}")
            beep()
        elif stampede and not last_stampede_state:
            send_alert(f"🚨 STAMPEDE ALERT! People={person_count}, Occ={occupancy_ratio:.2f}")
            beep()
        last_stampede_state = stampede

        # Draw boxes & labels
        color = (0,0,255) if stampede else (0,255,0)
        for tid,t in tracks.items():
            if t['missed']>0: continue
            x1,y1,x2,y2 = t['bbox']
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label = f"ID {tid}" + (f" | {t['speed']:.0f}px/s" if stampede else "")
            cv2.putText(frame,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        status = f"People {person_count} / Cap {max_capacity} | Occ {occupancy_ratio:.2f}"
        cv2.putText(frame,status,(16,32),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255) if stampede else (0,200,0),2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')

    client_socket.close()

# ===== Flask Routes =====
@app.route('/')
def index(): return render_template('index.html')

@app.route('/check_alert')
def check_alert(): return jsonify({'alert': ALERT_FLAG})

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session: return redirect(url_for('login'))
    max_capacity = session.get('max_capacity', MAX_CAPACITY)
    return Response(gen_frames(max_capacity), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== User Auth =====
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        name,email,password=request.form['name'],request.form['email'],request.form['password']
        hashed=generate_password_hash(password)
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        try:
            c.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",(name,email,hashed))
            conn.commit(); flash("Registration Successful! Please Login","success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists","danger")
        finally: conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        email,password=request.form['email'],request.form['password']
        conn=sqlite3.connect(DB_PATH); c=conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?",(email,))
        user=c.fetchone(); conn.close()
        if user and check_password_hash(user[3], password):
            session['user_id']=user[0]; session['user_name']=user[1]; flash("Login Successful","success")
            return redirect(url_for('predict'))
        else: flash("Invalid credentials","danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); flash("Logged out","info"); return redirect(url_for('index'))

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'user_id' not in session: return redirect(url_for('login'))
    if request.method=='POST':
        try:
            max_cap = int(request.form.get('max_capacity'))
            if max_cap<=0: raise ValueError
            session['max_capacity'] = max_cap
            flash(f"Max capacity set to {max_cap}. Starting camera...", "success")
        except:
            flash("Invalid capacity. Enter a positive number.", "danger")
    return render_template('predict.html', max_capacity=session.get('max_capacity', ''))

@app.route("/about")
def about(): return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0')
