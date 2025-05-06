import cv2
import time
import logging
import threading
import base64
import datetime

class DetectionSystem:
    """
    Main detection system that coordinates all components
    """
    def __init__(self, camera_manager, model_manager, detection_manager, alert_manager, 
                 heatmap_manager, db_manager, socketio, config):
        # Managers
        self.camera_manager = camera_manager
        self.model_manager = model_manager
        self.detection_manager = detection_manager
        self.alert_manager = alert_manager
        self.heatmap_manager = heatmap_manager
        self.db_manager = db_manager
        self.socketio = socketio
        
        # Configuration
        self.config = config
        self.camera_id = camera_manager.camera_id
        self.camera_url = camera_manager.camera_url
        self.process_every_n_frames = config['PROCESS_EVERY_N_FRAMES']
        
        # System state
        self.is_running = False
        self.processing = False
        self.frame_count = 0
        self.latest_processed_frame = None
        
        # Client tracking
        self.active_clients = 0
        self.heatmap_clients = 0
        
        # Initialize tracking variables
        self.last_emit_time = 0
        self.emit_interval = 0.1  # Emit frames every 100ms
        
        # Initialize logging
        logging.info(f"Detection system initialized for camera {self.camera_id}")
    
    def start(self):
        """Start the detection system"""
        if self.is_running:
            logging.info(f"Detection system for camera {self.camera_id} already running")
            return
            
        self.is_running = True
        
        # Get the socket handler from socketio
        socket_handler = None
        for attr in dir(self.socketio):
            if attr.endswith('handler'):
                handler = getattr(self.socketio, attr)
                if hasattr(handler, 'emit_frame'):
                    socket_handler = handler
                    break
                    
        # Set the socket handler for the camera manager if found
        if socket_handler:
            self.camera_manager.set_socket_handler(socket_handler)
            logging.info(f"Socket handler set for camera {self.camera_id}")
        else:
            logging.warning(f"No socket handler found for camera {self.camera_id}")
        
        # Start camera manager
        self.camera_manager.start()
        
        # Start processing thread
        processing_thread = threading.Thread(target=self._processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start frame emission thread
        emission_thread = threading.Thread(target=self._emission_loop)
        emission_thread.daemon = True
        emission_thread.start()
        
        logging.info(f"Detection system started for camera {self.camera_id}")
    
    def stop(self):
        """Stop the detection system"""
        self.is_running = False
        self.camera_manager.stop()
        logging.info(f"Detection system stopped for camera {self.camera_id}")

    @property
    def db_path(self):
        """Get the database path for this detection system"""
        return self.db_manager.db_path
        
    def update_client_count(self, video_count, heatmap_count):
        """Update the count of connected clients"""
        old_active = self.active_clients
        self.active_clients = video_count
        self.heatmap_clients = heatmap_count
        
        # Only log if the counts have changed
        if old_active != video_count:
            logging.info(f"Active clients for camera {self.camera_id}: video={video_count}, heatmap={heatmap_count}")
    
    def _emission_loop(self):
        """Dedicated loop for emitting frames to clients"""
        while self.is_running:
            try:
                if self.active_clients <= 0:
                    time.sleep(0.1)  # Sleep if no clients
                    continue
                    
                current_time = time.time()
                if current_time - self.last_emit_time < self.emit_interval:
                    time.sleep(0.01)  # Rate limiting
                    continue
                
                # Get the latest frame, which might be the processed one or just raw
                frame = self.latest_processed_frame if self.latest_processed_frame is not None else self.camera_manager.get_frame()
                
                if frame is not None:
                    self._emit_frame_update(frame)
                    self.last_emit_time = current_time
                
                time.sleep(0.01)  # Short sleep to avoid busy waiting
                
            except Exception as e:
                logging.error(f"Error in emission loop: {str(e)}")
                time.sleep(0.1)
    
    def _processing_loop(self):
        """Main processing loop for object detection"""
        last_flood_check_time = 0
        flood_check_interval = 10  # Check for flooding every 10 seconds
        last_heatmap_update = 0
        heatmap_update_interval = 5  # Update heatmap every 5 seconds
        
        while self.is_running:
            try:
                # Get next frame for processing
                frame = self.camera_manager.get_next_frame_for_processing(timeout=0.1)
                
                if frame is None:
                    time.sleep(0.01)  # Short sleep to avoid busy waiting
                    continue
                
                # Increment frame counter
                self.frame_count += 1
                
                # Only process every N frames for performance, but always keep the latest frame
                if self.frame_count % self.process_every_n_frames == 0:
                    self._process_frame(frame)
                
                # Check for flood periodically
                current_time = time.time()
                if current_time - last_flood_check_time > flood_check_interval:
                    self._check_for_flood(frame)
                    last_flood_check_time = current_time
                
                # Update heatmap visualization for clients
                if self.heatmap_clients > 0 and current_time - last_heatmap_update > heatmap_update_interval:
                    if hasattr(self.heatmap_manager, 'emit_heatmap_data'):
                        self.heatmap_manager.emit_heatmap_data(self.camera_id)
                    last_heatmap_update = current_time
                
            except Exception as e:
                logging.error(f"Error in processing loop: {str(e)}")
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        """Process a single frame for object detection"""
        if self.processing:
            return
            
        try:
            self.processing = True
            
            # 1. Run object detection
            detections = self.detection_manager.detect_objects(frame)
            
            # 2. Update heatmap for detected objects
            for det in detections:
                if 'normalized_bbox' in det:
                    self.heatmap_manager.update_heatmap(
                        det['normalized_bbox'], 
                        self.camera_id
                    )
            
            # 3. Process animal detections and create alert if needed
            if detections:
                self.alert_manager.process_animal_detection(frame, detections, self.camera_id)
            
            # 4. Annotate frame with detection results
            self.latest_processed_frame = self.detection_manager.annotate_frame(frame, detections)
            
            self.processing = False
            
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            self.processing = False
    
    def _check_for_flood(self, frame):
        """Check for potential flooding in the frame"""
        try:
            flood_confidence = self.detection_manager.detect_flood(frame)
            
            if flood_confidence > 0:
                # Create flood alert
                self.alert_manager.process_flood_detection(
                    frame, flood_confidence, self.camera_id
                )
                
                # Update the latest processed frame with flood annotation if exists
                if self.latest_processed_frame is not None:
                    self.latest_processed_frame = self.detection_manager.annotate_frame(
                        self.latest_processed_frame, [], flood_confidence
                    )
        
        except Exception as e:
            logging.error(f"Error checking for flood: {str(e)}")
    
    def _emit_frame_update(self, frame):
        """Send the latest processed frame to connected clients"""
        try:
            # For direct use of socketio
            if hasattr(self.socketio, 'emit'):
                # Resize frame before encoding to reduce network load
                small_frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Send the frame via WebSocket
                self.socketio.emit('frame_update', {
                    'image': jpg_as_text,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'camera_id': self.camera_id
                })
                
            # Log every 300 frames to avoid excessive logging
            if self.frame_count % 300 == 0:
                logging.info(f"Frame {self.frame_count} emitted for camera {self.camera_id}")
            
        except Exception as e:
            logging.error(f"Error emitting frame update: {str(e)}")