# main.py - updated with eventlet integration

import os
import logging
import threading
import time
import json
import eventlet
# Apply monkey patching early
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv

# Import configuration
from config import *

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename=LOG_FILE,
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variables to track detection systems
detection_systems = {}

def initialize_app():
    """Initialize the Flask application and SocketIO"""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    
    # Use eventlet for async mode
    async_mode = 'eventlet'
    logging.info("Using eventlet for async mode")
    
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*", 
        async_mode=async_mode,
        ping_timeout=60,
        ping_interval=25,
        logger=True,
        engineio_logger=FLASK_DEBUG
    )
    
    # Import modules here to avoid circular imports
    from api.routes import register_routes
    from api.socket_events import SocketEventHandler
    from modules.db_manager import DatabaseManager
    
    # Initialize database
    db_manager = DatabaseManager(DB_PATH)
    
    # Initialize socket events handler
    socket_handler = SocketEventHandler(socketio, detection_systems)
    
    # Register API routes
    register_routes(app, detection_systems)
    
    return app, socketio, socket_handler, db_manager

def load_camera_config():
    """Load camera configuration from config file or environment variables"""
    # First try loading from a JSON config file if it exists
    config_file = os.getenv('CAMERA_CONFIG', 'camera_config.json')
    cameras = []
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                cameras = json.load(f)
            logging.info(f"Loaded {len(cameras)} cameras from config file")
        except Exception as e:
            logging.error(f"Error loading camera config file: {str(e)}")
    
    # If no cameras loaded from file, use config.py definitions
    if not cameras:
        for camera_id, camera_url in CAMERA_URL.items():
            if camera_url and camera_url != "None":
                camera_config = {
                    'camera_id': camera_id,
                    'camera_url': camera_url,
                    'detection_threshold': DETECTION_THRESHOLD
                }
                cameras.append(camera_config)
        
        if cameras:
            logging.info(f"Loaded {len(cameras)} cameras from config.py")
        else:
            # Fall back to using default camera
            default_camera = {
                'camera_id': DEFAULT_CAMERA_ID,
                'camera_url': CAMERA_URL[DEFAULT_CAMERA_ID],
                'detection_threshold': DETECTION_THRESHOLD
            }
            cameras = [default_camera]
            logging.info("Using default camera configuration")
    
    return cameras

def initialize_detection_systems(camera_configs, socketio, db_manager):
    """Initialize all components and detection systems for each camera"""
    # Import required modules here to avoid circular imports
    from modules.camera_manager import CameraManager
    from modules.model_manager import ModelManager
    from modules.detection_manager import DetectionManager
    from modules.alert_manager import AlertManager
    from modules.heatmap_manager import HeatmapManager
    from modules.detection_system import DetectionSystem
    from utils.image_processing import ImageProcessor  # Import from utils folder
    
    for cam_config in camera_configs:
        camera_id = cam_config['camera_id']
        try:
            # Initialize image processor
            image_processor = ImageProcessor(
                process_width=PROCESS_WIDTH,
                process_height=PROCESS_HEIGHT,
                heatmap_decay=HEATMAP_DECAY
            )
            
            # Initialize all managers for this camera
            camera_manager = CameraManager(
                camera_url=cam_config['camera_url'],
                camera_id=camera_id
            )
            
            model_manager = ModelManager(
                model_path=cam_config.get('model_path'),
                detection_threshold=cam_config.get('detection_threshold', DETECTION_THRESHOLD)
            )
            
            detection_manager = DetectionManager(
                model_manager,
                image_processor
            )
            
            heatmap_manager = HeatmapManager(
                db_manager,
                socketio,
                image_processor
            )
            
            alert_manager = AlertManager(
                db_manager,
                socketio,
                globals() # Pass config
            )
            
            # Create detection system for this camera
            detection_system = DetectionSystem(
                camera_manager=camera_manager,
                model_manager=model_manager,
                detection_manager=detection_manager,
                alert_manager=alert_manager,
                heatmap_manager=heatmap_manager,
                db_manager=db_manager,
                socketio=socketio,
                config=globals()  # Pass all config variables
            )
            
            # Give the socket handler to the camera manager
            if hasattr(socketio, 'socket_handler'):
                camera_manager.set_socket_handler(socketio.socket_handler)
            
            detection_systems[camera_id] = detection_system
            logging.info(f"Initialized detection system for camera {camera_id}")
            
        except Exception as e:
            logging.error(f"Failed to initialize detection system for camera {camera_id}: {str(e)}")

def start_detection_systems():
    """Start all detection systems"""
    for camera_id, system in detection_systems.items():
        try:
            system.start()
            logging.info(f"Started detection system for camera {camera_id}")
        except Exception as e:
            logging.error(f"Failed to start detection system for camera {camera_id}: {str(e)}")

def monitor_connections(socket_handler):
    """Monitor socket connections periodically"""
    while True:
        try:
            connected_clients = len(socket_handler.connected_clients)
            video_subscribers = len(socket_handler.video_subscribers)
            
            logging.info(f"Connected clients: {connected_clients}, Video subscribers: {video_subscribers}")
            
            # Update client counts for each detection system
            for camera_id, system in detection_systems.items():
                system.update_client_count(video_subscribers, len(socket_handler.heatmap_subscribers))
                
            # Use eventlet.sleep instead of time.sleep
            eventlet.sleep(10)  # Check every 10 seconds
        except Exception as e:
            logging.error(f"Error in monitor_connections: {str(e)}")
            eventlet.sleep(10)

def main():
    """Main application entry point"""
    try:
        # Initialize app and socketio
        app, socketio, socket_handler, db_manager = initialize_app() 
        
        # Configure GPU memory growth to avoid OOM errors
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except Exception as e:
            logging.warning(f"GPU configuration warning: {str(e)}")
        
        # Load camera configurations
        camera_configs = load_camera_config()
        
        # Initialize detection systems
        initialize_detection_systems(camera_configs, socketio, db_manager)
        
        # Start detection systems
        start_detection_systems()
        
        # Start the monitoring thread for connection tracking - use eventlet spawn
        eventlet.spawn(monitor_connections, socket_handler)
        logging.info("Monitor connections thread started")
        
        # Run the Flask app with SocketIO
        logging.info(f"Starting server on {FLASK_HOST}:{FLASK_PORT}")
        socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
        
    except Exception as e:
        logging.critical(f"Critical error in main: {str(e)}")
        # Exit with error code
        exit(1)

if __name__ == "__main__":
    main()