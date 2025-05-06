import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Camera settings
CAMERA_URL = {
    'cam1': os.getenv('CAMERA_URL_1', 'anjing3.mp4'),
    'cam2': os.getenv('CAMERA_URL_2', 'camera2_default.mp4'),
    'cam3': os.getenv('CAMERA_URL_3', 'camera3_default.mp4'),
    'cam4': os.getenv('CAMERA_URL_4', 'camera4_default.mp4')
}
DEFAULT_CAMERA_ID = os.getenv('DEFAULT_CAMERA_ID', 'cam1')

# Database settings
DB_PATH = os.getenv('DB_PATH', 'detections.db')

# Detection settings
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', 0.6))
PROCESS_EVERY_N_FRAMES = int(os.getenv('PROCESS_EVERY_N_FRAMES', 5))
PROCESS_WIDTH = int(os.getenv('PROCESS_WIDTH', 320))
PROCESS_HEIGHT = int(os.getenv('PROCESS_HEIGHT', 240))
ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', 300))  # 5 minutes

# Animal classes for detection (SSD MobileNet V2 COCO)
# 17: cat, 18: dog, 19: horse, 20: sheep, 21: cow
ANIMAL_CLASSES = [int(x) for x in os.getenv('ANIMAL_CLASSES', '17,18,19,20,21').split(',')]
CLASS_NAMES = {
    17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow'
}

# Heatmap settings
HEATMAP_DECAY = float(os.getenv('HEATMAP_DECAY', 0.99))
HEATMAP_UPDATE_INTERVAL = float(os.getenv('HEATMAP_UPDATE_INTERVAL', 2.0))

# Flask settings
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

# Logging settings
LOG_FILE = os.getenv('LOG_FILE', 'detection_logs.log')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')