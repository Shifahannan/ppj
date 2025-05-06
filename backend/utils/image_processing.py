# image_processing.py
# Image processing utilities for the detection system

import cv2
import numpy as np
import logging
import time
import tensorflow as tf
import random

class ImageProcessor:
    def __init__(self, process_width=320, process_height=240, heatmap_decay=0.99):
        """
        Initialize image processor
        
        Args:
            process_width: Width to resize frames to for processing
            process_height: Height to resize frames to for processing
            heatmap_decay: Decay factor for heatmap intensity over time
        """
        self.process_width = process_width
        self.process_height = process_height
        self.heatmap_decay = heatmap_decay
        self.heatmap_data = np.zeros((process_height, process_width), dtype=np.float32)
        self.normalized_heatmap = None
        
        # For performance metrics
        self.last_processing_time = 0
        self.processing_times = []  # Store last 100 processing times
        
        logging.info(f"ImageProcessor initialized with dimensions {process_width}x{process_height}")
    
    def resize_for_processing(self, frame):
        """Resize frame to standard processing size"""
        return cv2.resize(frame, (self.process_width, self.process_height))
    
    def resize_for_display(self, frame, width=640, height=480):
        """Resize frame for display/streaming"""
        return cv2.resize(frame, (width, height))
    
    def add_timestamp(self, frame):
        """Add timestamp to frame"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame
    
    def draw_detections(self, frame, detections, class_names):
        """
        Draw bounding boxes and labels for detected objects
        
        Args:
            frame: Input image frame
            detections: List of detection objects with class, confidence, and bbox
            class_names: Dictionary mapping class IDs to names
        
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            class_id = detection['class']
            confidence = detection['confidence']
            
            # Get class name
            class_name = class_names.get(class_id, 'unknown')
            
            # Draw rectangle
            cv2.rectangle(output_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(output_frame, label, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return output_frame
    
    def draw_flood_warning(self, frame, confidence):
        """Add flood warning text to frame"""
        height = frame.shape[0]
        cv2.putText(frame, f"FLOOD RISK: {confidence:.2f}", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame
    
    def update_heatmap(self, bbox):
        """
        Update heatmap data based on detection
        
        Args:
            bbox: Bounding box (xmin, ymin, xmax, ymax)
        """
        try:
            # Apply decay to existing heatmap
            self.heatmap_data *= self.heatmap_decay
            
            # Extract center of bounding box
            xmin, ymin, xmax, ymax = bbox
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            
            # Make sure the center point is within bounds
            if 0 <= center_x < self.heatmap_data.shape[1] and 0 <= center_y < self.heatmap_data.shape[0]:
                # Add detection to heatmap with Gaussian distribution
                sigma = 5  # Smaller sigma for better performance
                x_range = min(10, self.heatmap_data.shape[1] - center_x, center_x)
                y_range = min(10, self.heatmap_data.shape[0] - center_y, center_y)
                
                for x in range(center_x - x_range, center_x + x_range):
                    for y in range(center_y - y_range, center_y + y_range):
                        if 0 <= x < self.heatmap_data.shape[1] and 0 <= y < self.heatmap_data.shape[0]:
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            intensity = np.exp(-(dist**2) / (2 * sigma**2))
                            self.heatmap_data[y, x] += intensity
                
                # Set flag to regenerate normalized heatmap
                self.normalized_heatmap = None
                
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error updating heatmap: {str(e)}")
            return False
    
    def get_normalized_heatmap(self):
        """Get a normalized and colorized version of the heatmap"""
        if self.normalized_heatmap is None:
            try:
                # Normalize heatmap to 0-255 range
                normalized = cv2.normalize(self.heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
                normalized = normalized.astype(np.uint8)
                
                # Apply colormap
                colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                
                # Store the normalized heatmap
                self.normalized_heatmap = colorized
            except Exception as e:
                logging.error(f"Error normalizing heatmap: {str(e)}")
                # Return empty heatmap on error
                self.normalized_heatmap = np.zeros((self.process_height, self.process_width, 3), dtype=np.uint8)
        
        return self.normalized_heatmap
    
    def sample_heatmap_points(self, threshold=50, max_points=200):
        """
        Extract points from heatmap for visualization
        
        Args:
            threshold: Only include points with intensity > threshold
            max_points: Maximum number of points to return
        
        Returns:
            List of {x, y, intensity} points normalized to 0-1 range
        """
        points = []
        for y in range(self.process_height):
            for x in range(self.process_width):
                intensity = int(self.heatmap_data[y, x] * 255)
                if intensity > threshold:
                    # Normalize to 0-1 for frontend
                    points.append({
                        'x': x / self.process_width,
                        'y': y / self.process_height,
                        'intensity': intensity / 255.0
                    })
        
        # Sample points if too many
        if len(points) > max_points:
            return random.sample(points, max_points)
        
        return points
    
    def detect_flood(self, frame):
        """
        Detect potential flooding in the frame
        
        Args:
            frame: CV2 image frame
        
        Returns:
            Flood detection confidence (0-1)
        """
        try:
            # Resize for faster processing
            small_frame = self.resize_for_processing(frame)
            
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
    
    def measure_performance(self, start_time):
        """
        Measure and track image processing performance
        
        Args:
            start_time: Start time of processing
        
        Returns:
            Processing time in milliseconds
        """
        process_time = (time.time() - start_time) * 1000  # Convert to ms
        self.processing_times.append(process_time)
        
        # Keep only the last 100 measurements
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return process_time
    
    def get_average_processing_time(self):
        """Get average processing time from recent frames"""
        if not self.processing_times:
            return 0
        return sum(self.processing_times) / len(self.processing_times)