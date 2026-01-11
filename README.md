# HUSH RUSH  
## AI-Based Pre-Stampede Prevention and Alert System

## Project Overview
Stampede incidents in crowded public places such as temples, festivals, railway stations, and stadiums pose serious risks to human life. Traditional CCTV surveillance systems only record video and lack real-time intelligence to analyze crowd behavior or predict dangerous situations.

HUSH RUSH is an AI-Based Pre-Stampede Prevention and Alert System that uses computer vision and deep learning to monitor crowd density and movement patterns in real time. The system detects potential stampede conditions at an early stage and automatically generates alerts to enable timely preventive action.

---

## Objectives
- To monitor crowd density in real time
- To detect abnormal crowd movement patterns
- To predict potential stampede situations
- To generate instant alerts for authorities
- To enhance public safety using AI-based surveillance

---

## Key Features
- Real-time video streaming using socket programming
- AI-based person detection using YOLOv5
- Crowd counting and occupancy ratio calculation
- Centroid-based movement tracking
- Abnormal speed detection for panic behavior
- Automated alerts via SMS, audio alarm, and web dashboard
- Secure user authentication system

---

## System Architecture
1. Raspberry Pi camera captures live video
2. Video frames are transmitted using socket communication
3. YOLOv5 analyzes frames and detects people
4. Crowd density and movement speed are calculated
5. Stampede risk is identified using threshold logic
6. Alerts are generated and displayed on a web interface

---

## Technologies Used

### Hardware
- Raspberry Pi
- Camera Module

### Software
- Python
- OpenCV
- YOLOv5
- PyTorch
- Flask
- SQLite

### APIs & Tools
- Twilio API (SMS alerts)
- Socket Programming

---

## Methodology
- Capture live video from Raspberry Pi camera
- Stream video frames to the processing system
- Apply YOLOv5 for real-time person detection
- Track individuals using centroid-based tracking
- Calculate crowd density and movement speed
- Detect stampede conditions based on thresholds
- Generate alerts automatically when risk is detected

---

## Alert Mechanism
The system provides alerts through:
- SMS notifications to registered mobile numbers
- Audio alerts for immediate warning
- Web-based dashboard notifications

---

## Results
- Accurate detection of people in crowded environments
- Effective identification of overcrowding situations
- Successful detection of abnormal crowd movement
- Timely alert generation before critical situations
- Real-time monitoring through a web interface

---

## Limitations
- Detection accuracy depends on camera quality and lighting
- Performance may degrade in extreme crowd occlusion
- Large-scale deployment may require GPU support

---

## Future Enhancements
- Multi-camera integration
- Crowd heatmap visualization
- Edge AI deployment on Raspberry Pi
- Cloud-based monitoring and analytics
- Integration with emergency response systems

---

## How to Run the Project
1. Clone the repository
2. Install required dependencies  
   pip install -r requirements.txt
3. Start the Raspberry Pi camera server
4. Run the Flask application  
   python app.py
5. Open the browser and access the dashboard

---

## Conclusion
HUSH RUSH demonstrates how artificial intelligence and computer vision can be effectively used to improve public safety. By providing real-time crowd analysis and early warnings, the system significantly reduces the risk of stampede incidents and supports smarter crowd management.

---


