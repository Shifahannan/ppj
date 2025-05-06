# heatmap_manager.py
import cv2
import numpy as np
import logging
import time
import base64
import datetime

class HeatmapManager:
    def __init__(self, config, db_manager):
        """Initialize the heatmap manager"""
        self.config = config
        self.db_manager = db_manager
        
        # Initialize heatmap data structure
        self.width = config['camera']['process_width']
        self.height = config['camera']['process_height']
        self.heatmap_data = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Heatmap parameters
        self.decay = config['heatmap']['decay']
        self.update_interval = config['heatmap']['update_interval']
        self.sigma = config['heatmap']['sigma']
        self.db_sample_rate = config['database']['sample_rate']
        
        # Cached normalized heatmap
        self.normalized_heatmap = None
        self.last_update_time = time.time()
        
        logging.info("Heatmap manager initialized")
    
    def update(self, detection):
        """
        Update heatmap with a new detection
        
        Args:
            detection: Dictionary with detection information
        """
        try:
            # Apply decay to existing heatmap
            self.heatmap_data *= self.decay
            
            # Extract bounding box
            if 'bbox_norm' in detection:
                # Use normalized coordinates
                xmin, ymin, xmax, ymax = detection['bbox_norm']
                center_x = int((xmin + xmax) / 2 * self.width)
                center_y = int((ymin + ymax) / 2 * self.height)
            else:
                # Use pixel coordinates and normalize to heatmap size
                xmin, ymin, xmax, ymax = detection['bbox']
                # Get original frame dimensions
                orig_height, orig_width = detection.get('frame_size', (self.height, self.width))
                # Scale to heatmap dimensions
                center_x = int(((xmin + xmax) / 2 / orig_width) * self.width)
                center_y = int(((ymin + ymax) / 2 / orig_height) * self.height)
            
            # Make sure the center point is within bounds
            if 0 <= center_x < self.width and 0 <= center_y < self.height:
                # Add detection to heatmap with Gaussian distribution
                self._add_gaussian_blob(center_x, center_y)
                
                # Store in database with sampling for efficiency
                if np.random.random() < self.db_sample_rate:
                    try:
                        self.db_manager.store_heatmap_point(
                            center_x, center_y, 1.0, detection.get('camera_id', 'default')
                        )
                    except Exception as e:
                        logging.error(f"Error storing heatmap point: {str(e)}")
                
                # Reset cached normalized heatmap
                self.normalized_heatmap = None
        except Exception as e:
            logging.error(f"Error updating heatmap: {str(e)}")
    
    def _add_gaussian_blob(self, center_x, center_y):
        """Add a Gaussian blob to the heatmap at the specified center"""
        sigma = self.sigma
        range_factor = 2  # How many sigmas to include in each direction
        
        # Calculate blob boundaries
        x_min = max(0, center_x - int(sigma * range_factor))
        x_max = min(self.width, center_x + int(sigma * range_factor) + 1)
        y_min = max(0, center_y - int(sigma * range_factor))
        y_max = min(self.height, center_y + int(sigma * range_factor) + 1)
        
        # Optimization: Only update pixels within this range
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                intensity = np.exp(-dist_sq / (2 * sigma**2))
                self.heatmap_data[y, x] += intensity
    
    def get_normalized_heatmap(self):
        """Get a normalized and colorized version of the heatmap"""
        if self.normalized_heatmap is None:
            try:
                # Check if there's any data in the heatmap
                if np.max(self.heatmap_data) > 0:
                    # Normalize to 0-255 range
                    normalized = cv2.normalize(self.heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
                    normalized = normalized.astype(np.uint8)
                    
                    # Apply colormap
                    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                else:
                    # If no data, return empty heatmap
                    colorized = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                self.normalized_heatmap = colorized
            except Exception as e:
                logging.error(f"Error normalizing heatmap: {str(e)}")
                # Return empty heatmap on error
                self.normalized_heatmap = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        return self.normalized_heatmap
    
    def get_heatmap_data_for_websocket(self):
        """Prepare heatmap data for WebSocket transmission"""
        current_time = time.time()
        
        # Check if it's time to update
        if current_time - self.last_update_time < self.update_interval:
            return None
        
        self.last_update_time = current_time
        
        try:
            # Get normalized heatmap
            heatmap = self.get_normalized_heatmap()
            
            # Encode heatmap to JPEG
            _, buffer = cv2.imencode('.jpg', heatmap, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Extract significant points for the frontend
            threshold = 50  # Only include points with intensity > threshold
            points = []
            
            # Use a more efficient approach to find significant points
            # Only sample a portion of the heatmap for performance
            sample_step = 4  # Sample every 4th pixel
            y_indices, x_indices = np.where(self.heatmap_data[::sample_step, ::sample_step] * 255 > threshold)
            
            # Convert sampled indices back to full heatmap coordinates
            for i in range(min(200, len(y_indices))):  # Limit to 200 points
                y = y_indices[i] * sample_step
                x = x_indices[i] * sample_step
                intensity = self.heatmap_data[y, x]
                
                points.append({
                    'x': float(x) / self.width,
                    'y': float(y) / self.height,
                    'intensity': float(intensity / np.max(self.heatmap_data) if np.max(self.heatmap_data) > 0 else 0)
                })
            
            return {
                'image': jpg_as_text,
                'points': points,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            logging.error(f"Error preparing heatmap data: {str(e)}")
            return None
    
    def reset(self):
        """Reset the heatmap data"""
        self.heatmap_data = np.zeros((self.height, self.width), dtype=np.float32)
        self.normalized_heatmap = None
        logging.info("Heatmap data reset")