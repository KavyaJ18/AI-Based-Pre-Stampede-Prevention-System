# pi_camera_server.py
import cv2
import socket
import struct
import pickle

# IP and port of the Raspberry Pi
HOST = '10.92.156.139'  # empty string means all available interfaces
PORT = 9992

# Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)
print("Waiting for connection...")

conn, addr = server_socket.accept()
print(f"Connected to: {addr}")

# Open Pi camera
cap = cv2.VideoCapture(0)  # Use 0 or the camera ID

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Serialize frame
    data = pickle.dumps(frame)
    # Send message length first
    message_size = struct.pack("L", len(data))
    conn.sendall(message_size + data)

cap.release()
conn.close()
server_socket.close()