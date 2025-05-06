import cv2
import numpy as np
import time
import logging
import datetime
from modules.model_manager import ModelManager  # Assuming you already have this

class DetectionManager:
    """
    Handles object detection and flood detection logic
    """
    def __init__(self, model_manager, config):
        self.model_manager = model_manager
        self.config = config
        self.detection_threshold = config.DETECTION_THRESHOLD
        self.animal_classes = config.ANIMAL_CLASSES
        self.class_names = config.CLASS_NAMES
        self.process_width = config.PROCESS_WIDTH
        self.process_height = config.PROCESS_HEIGHT
        
        # Performance tracking
        self.last_detection_time = 0
        self.detection_count = 0
        self.avg_detection_time = 0
        
    def detect_objects(self, frame):
        """
        Detect objects (animals) in a frame
        
        Args:
            frame: CV2 image frame
        
        Returns:
            List of detected objects with class, confidence and bounding box
        """
        start_time = time.time()
        
        try:
            # Resize frame to improve performance
            resized_frame = cv2.resize(frame, (self.process_width, self.process_height))
            
            # Convert to RGB (TensorFlow models expect RGB)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # Get detections from model manager
            detections = self.model_manager.detect(rgb_frame)
            
            # Process results
            detected_objects = []
            orig_height, orig_width, _ = frame.shape
            
            # Handle different model output formats
            if isinstance(detections, dict) and 'detection_boxes' in detections:
                boxes = detections['detection_boxes'][0].numpy()
                classes = detections['detection_classes'][0].numpy().astype(np.int32)
                scores = detections['detection_scores'][0].numpy()
                
                # Filter detections
                for i in range(min(20, len(scores))):  # Limit to top 20 detections for performance
                    if scores[i] > self.detection_threshold:
                        if classes[i] in self.animal_classes:
                            ymin, xmin, ymax, xmax = boxes[i]
                            
                            # Convert normalized coordinates to original frame size
                            xmin_orig = int(xmin * orig_width)
                            ymin_orig = int(ymin * orig_height)
                            xmax_orig = int(xmax * orig_width)
                            ymax_orig = int(ymax * orig_height)
                            
                            bbox = (xmin_orig, ymin_orig, xmax_orig, ymax_orig)
                            
                            # Add to detected objects
                            detected_objects.append({
                                'class': classes[i],
                                'class_name': self.class_names.get(classes[i], 'unknown'),
                                'confidence': float(scores[i]),
                                'bbox': bbox,
                                'normalized_bbox': (
                                    float(xmin), float(ymin), 
                                    float(xmax), float(ymax)
                                )
                            })
            
            # Update performance metrics
            end_time = time.time()
            detection_time = end_time - start_time
            self.detection_count += 1
            self.avg_detection_time = (self.avg_detection_time * (self.detection_count - 1) + detection_time) / self.detection_count
            self.last_detection_time = detection_time
            
            return detected_objects
        
        except Exception as e:
            logging.error(f"Error in object detection: {str(e)}")
            return []
    
    def detect_flood(self, frame):
        """
        Detect potential flooding in the frame using color analysis
        
        Args:
            frame: CV2 image frame
        
        Returns:
            Flood detection confidence (0-1)
        """
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (self.process_width, self.process_height))
            
            # Convert to HSV for better water detection
            hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
            
            # Water typically has these HSV ranges
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue-ish colors (potential water)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Calculate percentage of frame covered by potential water
            water_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            water_ratio = water_pixels / total_pixels
            
            # If water covers more than 30% of the frame, it might indicate flooding
            return water_ratio if water_ratio > 0.3 else 0
        
        except Exception as e:
            logging.error(f"Error in flood detection: {str(e)}")
            return 0
    
    def annotate_frame(self, frame, detections, flood_confidence=0):
        """
        Add visual annotations to the frame based on detections
        
        Args:
            frame: Original frame
            detections: List of detection objects
            flood_confidence: Flood detection confidence
            
        Returns:
            Annotated frame
        """
        try:
            output_frame = frame.copy()
            height, width, _ = frame.shape
            
            # Draw bounding boxes for detected animals
            for det in detections:
                xmin, ymin, xmax, ymax = det['bbox']
                class_name = det['class_name']
                confidence = det['confidence']
                
                # Draw rectangle
                cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(output_frame, label, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add flood detection indicator if present
            if flood_confidence > 0:
                cv2.putText(output_frame, f"FLOOD RISK: {flood_confidence:.2f}", 
                            (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(output_frame, timestamp, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return output_frame
        
        except Exception as e:
            logging.error(f"Error annotating frame: {str(e)}")
            return frame  # Return original frame on error
    
    def get_performance_stats(self):
        """Get detection performance statistics"""
        return {
            'avg_detection_time': self.avg_detection_time,
            'detection_count': self.detection_count,
            'last_detection_time': self.last_detection_time
        }