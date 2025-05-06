import cv2
import time
import logging
import datetime
import base64
import json
import sqlite3
from flask_socketio import SocketIO

class AlertManager:
    """
    Handles alert generation, storage, and notification
    """
    def __init__(self, db_manager, socketio, config):
        self.db_manager = db_manager
        self.socketio = socketio
        self.config = config
        self.alert_cooldown = config.ALERT_COOLDOWN
        
        # Track last alert times to prevent alert flooding
        self.last_alerts = {
            'animal': 0,
            'flood': 0
        }
        
        # Track statistics
        self.alert_counts = {
            'animal': 0,
            'flood': 0
        }
    
    def check_cooldown(self, alert_type):
        """
        Check if an alert type is in cooldown period
        
        Args:
            alert_type: Type of alert ('animal' or 'flood')
            
        Returns:
            True if alert can be sent, False if in cooldown
        """
        current_time = time.time()
        if alert_type in self.last_alerts:
            if current_time - self.last_alerts[alert_type] < self.alert_cooldown:
                return False
        return True
    
    def process_animal_detection(self, frame, detections, camera_id):
        """
        Process animal detections and create alert if needed
        
        Args:
            frame: Image frame
            detections: List of animal detections
            camera_id: Camera identifier
            
        Returns:
            True if alert was generated, False otherwise
        """
        if not detections or not self.check_cooldown('animal'):
            return False
        
        try:
            # Create alert details
            animal_types = [det['class_name'] for det in detections]
            animal_types_count = {}
            for animal in animal_types:
                if animal in animal_types_count:
                    animal_types_count[animal] += 1
                else:
                    animal_types_count[animal] = 1
                    
            # Format the details string
            animal_details = []
            for animal, count in animal_types_count.items():
                animal_details.append(f"{count} {animal}{'s' if count > 1 else ''}")
            
            details = f"Detected {len(detections)} stray animals: {', '.join(animal_details)}"
            avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
            
            # Store the alert
            self._store_alert('animal', frame, details, avg_confidence, camera_id)
            
            # Update tracking
            self.last_alerts['animal'] = time.time()
            self.alert_counts['animal'] += 1
            
            logging.info(f"Animal alert generated: {details}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing animal detection: {str(e)}")
            return False
    
    def process_flood_detection(self, frame, flood_confidence, camera_id):
        """
        Process flood detection and create alert if needed
        
        Args:
            frame: Image frame
            flood_confidence: Flood detection confidence
            camera_id: Camera identifier
            
        Returns:
            True if alert was generated, False otherwise
        """
        if flood_confidence <= 0 or not self.check_cooldown('flood'):
            return False
        
        try:
            details = f"Possible flooding detected with confidence {flood_confidence:.2f}"
            
            # Store the alert
            self._store_alert('flood', frame, details, flood_confidence, camera_id)
            
            # Update tracking
            self.last_alerts['flood'] = time.time()
            self.alert_counts['flood'] += 1
            
            logging.info(f"Flood alert generated: {details}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing flood detection: {str(e)}")
            return False
    
    def _store_alert(self, alert_type, frame, details, confidence, camera_id):
        """
        Store detection alert in database and send notification
        
        Args:
            alert_type: 'animal' or 'flood'
            frame: Image frame containing the detection
            details: Description of the detection
            confidence: Detection confidence
            camera_id: Camera identifier
        """
        try:
            # Resize the frame before encoding to reduce database size
            small_frame = cv2.resize(frame, (640, 480))

            # Convert image to JPEG format with lower quality
            _, img_encoded = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_bytes = img_encoded.tobytes()
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Store in database using db_manager
            alert_id = self.db_manager.store_alert(
                timestamp, alert_type, details, confidence, img_bytes, camera_id
            )
            
            # Emit WebSocket event
            alert_data = {
                'id': alert_id,
                'timestamp': timestamp,
                'type': alert_type,
                'details': details,
                'confidence': confidence,
                'camera_id': camera_id,
                'image': base64.b64encode(img_bytes).decode('utf-8')
            }
            
            self.socketio.emit('new_alert', alert_data)
            
        except Exception as e:
            logging.error(f"Failed to store alert: {str(e)}")
    
    def get_alert_stats(self):
        """Get alert statistics"""
        return {
            'counts': self.alert_counts,
            'last_alerts': self.last_alerts
        }