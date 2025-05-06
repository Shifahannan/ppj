import cv2
import time
import logging
import eventlet
import os
from queue import Queue

class CameraManager:
    """
    Handles camera operations, video capture, and frame processing
    Optimized for use with eventlet
    """
    def __init__(self, camera_url, camera_id, process_width=640, process_height=480, max_fps=30):
        self.camera_url = camera_url
        self.camera_id = camera_id
        self.process_width = process_width
        self.process_height = process_height
        self.max_fps = max_fps
        
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=5)  # Buffer for frames
        self.last_frame = None
        self.frame_count = 0
        self.socket_handler = None  # Will be set later
        
        self._initialize_camera()
        
    def _initialize_camera(self):
        """Initialize the camera capture"""
        try:
            logging.info(f"Initializing camera with URL: {self.camera_url}")
            
            # Check if it's a file path that needs to be resolved
            if not self.camera_url.startswith(('rtsp://', 'http://', 'https://', '/')):
                # Check if file exists in current directory
                if os.path.exists(self.camera_url):
                    absolute_path = os.path.abspath(self.camera_url)
                    logging.info(f"Using local video file: {absolute_path}")
                    self.cap = cv2.VideoCapture(absolute_path)
                else:
                    # Try to interpret as a device index (0, 1, etc.)
                    try:
                        device_index = int(self.camera_url)
                        logging.info(f"Using camera device index: {device_index}")
                        self.cap = cv2.VideoCapture(device_index)
                    except ValueError:
                        logging.error(f"Could not interpret camera URL: {self.camera_url}")
                        return False
            else:
                # It's a URL, use directly
                self.cap = cv2.VideoCapture(self.camera_url)
            
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera: {self.camera_url}")
                return False
                
            # Try to set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.process_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.process_height)
            
            # For RTSP streams, reduce buffer size
            if self.camera_url.startswith('rtsp://'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
            # Read a test frame to verify the camera works
            ret, test_frame = self.cap.read()
            if not ret:
                logging.error(f"Failed to read test frame from camera: {self.camera_url}")
                return False
                
            # Get actual frame dimensions
            actual_height, actual_width = test_frame.shape[:2]
            logging.info(f"Camera initialized successfully: {self.camera_id}, dimensions: {actual_width}x{actual_height}")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing camera {self.camera_id}: {str(e)}")
            return False
    
    def set_socket_handler(self, socket_handler):
        """Set the socket handler for frame emission"""
        self.socket_handler = socket_handler
        logging.info(f"Socket handler set for camera {self.camera_id}")
    
    def start(self):
        """Start the camera capture thread"""
        if self.is_running:
            logging.info(f"Camera {self.camera_id} already running")
            return
            
        if self.cap is None or not self.cap.isOpened():
            success = self._initialize_camera()
            if not success:
                logging.error(f"Failed to initialize camera {self.camera_id}, cannot start")
                return
                
        self.is_running = True
        # Use eventlet.spawn instead of threading
        eventlet.spawn(self._capture_loop)
        logging.info(f"Camera capture started for {self.camera_id}")
        
    def stop(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logging.info(f"Camera capture stopped for {self.camera_id}")
    
    def _capture_loop(self):
        """Main loop for capturing frames"""
        last_frame_time = time.time()
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        emit_counter = 0
        
        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logging.warning(f"Camera {self.camera_id} disconnected, attempting to reconnect")
                    eventlet.sleep(1)
                    reconnect_attempts += 1
                    
                    if reconnect_attempts > max_reconnect_attempts:
                        logging.error(f"Failed to reconnect to camera {self.camera_id} after {max_reconnect_attempts} attempts")
                        eventlet.sleep(5)  # Wait longer before retrying
                        reconnect_attempts = 0
                    
                    self._initialize_camera()
                    continue
                
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.warning(f"Failed to receive frame from camera {self.camera_id}")
                    
                    # Handle end of video file by looping
                    if self.camera_url.endswith(('.mp4', '.avi', '.mov')):
                        logging.info(f"End of video file reached for {self.camera_id}, restarting")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind the video
                    else:
                        eventlet.sleep(0.5)
                    continue
                
                # Reset reconnect attempts counter on successful frame read
                reconnect_attempts = 0
                
                # Rate limiting to avoid excessive CPU usage
                current_time = time.time()
                elapsed = current_time - last_frame_time
                min_frame_time = 1.0 / self.max_fps
                
                if elapsed < min_frame_time:
                    eventlet.sleep(min_frame_time - elapsed)
                
                # Update timing and frame count
                last_frame_time = time.time()
                self.frame_count += 1
                
                # Store the frame
                self.last_frame = frame.copy()
                
                # Add to queue for processing, but don't block if queue is full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Emit the frame directly if a socket handler is available
                # Only emit every few frames to reduce bandwidth
                emit_counter += 1
                if self.socket_handler and emit_counter % 2 == 0:  # Emit every 2nd frame
                    self.socket_handler.emit_frame(self.camera_id, frame)
                
                # Yield control to other greenlets
                eventlet.sleep(0)
                
            except Exception as e:
                logging.error(f"Error in camera capture loop for {self.camera_id}: {str(e)}")
                eventlet.sleep(1)
    
    def get_frame(self):
        """Get the latest frame (non-blocking)"""
        if self.last_frame is None:
            return None
        return self.last_frame.copy()
    
    def get_next_frame_for_processing(self, timeout=1):
        """Get next frame from queue for processing"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except:
            return None