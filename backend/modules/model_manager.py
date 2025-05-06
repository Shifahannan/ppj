# model_manager.py
import os
import logging
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import cv2  # Added import for cv2 which was missing but used in the code

# Load environment variables from .env file
load_dotenv()

class ModelManager:
    def __init__(self, config):
        """Initialize model manager with configuration"""
        self.config = config
        # Get model path from environment variable or config
        self.model_path = os.getenv('DETECTION_MODEL_PATH', config['detection']['model_path'])
        self.detection_model = None
        self.using_tflite = False
        self.interpreter = None
        
        # Setup GPU memory management
        self._configure_gpu()
        
        # Load model
        self._load_model()
    
    def _configure_gpu(self):
        """Configure GPU memory growth to avoid OOM errors"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s), memory growth enabled")
                
                # Get mixed precision policy from environment or use default
                mixed_precision = os.getenv('TF_MIXED_PRECISION', 'mixed_float16')
                tf.keras.mixed_precision.set_global_policy(mixed_precision)
                logging.info(f"Mixed precision set to {mixed_precision}")
            except RuntimeError as e:
                logging.error(f"GPU memory configuration error: {str(e)}")
    
    def _load_model(self):
        """Load detection model with appropriate format"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                self._load_custom_model()
            else:
                self._load_pretrained_model()
                
            # Test the model with a blank image to ensure it's working
            self._test_model()
            logging.info("Model loaded and tested successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            self._load_fallback_model()
    
    def _load_custom_model(self):
        """Load a custom model from the specified path"""
        try:
            if self.model_path.endswith('.tflite'):
                # Load as TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.using_tflite = True
                logging.info(f"Loaded TFLite model from {self.model_path}")
            else:
                # Load as SavedModel
                self.detection_model = tf.saved_model.load(self.model_path)
                self.using_tflite = False
                logging.info(f"Loaded TF SavedModel from {self.model_path}")
        except Exception as e:
            logging.error(f"Error loading custom model: {str(e)}")
            raise
    
    def _load_pretrained_model(self):
        """Load a pre-trained model from TensorFlow Hub"""
        try:
            import tensorflow_hub as hub
            # Get detector URL from environment variable or use default
            detector_url = os.getenv(
                'DETECTOR_URL', 
                "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v2/1/metadata/2"
            )
            self.detection_model = hub.load(detector_url)
            self.using_tflite = False
            logging.info(f"TensorFlow Hub model loaded successfully from {detector_url}")
        except Exception as e:
            logging.error(f"Error loading TF Hub model: {str(e)}")
            raise
    
    def _load_fallback_model(self):
        """Load a simpler fallback model if the main model fails"""
        try:
            # Get fallback model name from environment variable or use default
            fallback_model = os.getenv('FALLBACK_MODEL', 'MobileNetV2')
            
            if fallback_model == 'MobileNetV2':
                from tensorflow.keras.applications import MobileNetV2
                logging.warning("Using fallback classification model (MobileNetV2)")
                self.detection_model = MobileNetV2(weights='imagenet', include_top=True)
            elif fallback_model == 'ResNet50':
                from tensorflow.keras.applications import ResNet50
                logging.warning("Using fallback classification model (ResNet50)")
                self.detection_model = ResNet50(weights='imagenet', include_top=True)
            else:
                logging.warning(f"Unknown fallback model {fallback_model}, using MobileNetV2")
                from tensorflow.keras.applications import MobileNetV2
                self.detection_model = MobileNetV2(weights='imagenet', include_top=True)
                
            self.using_tflite = False
        except Exception as e:
            logging.error(f"Error loading fallback model: {str(e)}")
            # Create a dummy model that returns no detections
            self.detection_model = None
            logging.critical("No working model could be loaded!")
    
    def _test_model(self):
        """Test the model with a blank image to ensure it works"""
        try:
            # Get test dimensions from env vars or config
            process_height = int(os.getenv('PROCESS_HEIGHT', 
                                          self.config['camera']['process_height']))
            process_width = int(os.getenv('PROCESS_WIDTH', 
                                         self.config['camera']['process_width']))
            
            test_image = np.zeros((process_height, process_width, 3), dtype=np.uint8)
            _ = self.detect_objects(test_image)
        except Exception as e:
            logging.error(f"Model test failed: {str(e)}")
            raise
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame: CV2 image frame
        
        Returns:
            List of detected objects with class, confidence and bounding box
        """
        if self.detection_model is None and self.interpreter is None:
            return []  # No model available
        
        try:
            # Convert to RGB (TensorFlow models expect RGB)
            rgb_frame = frame
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = frame.copy()
                if frame.dtype == np.uint8:  # Only convert if it's a BGR image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process based on model type
            if self.using_tflite:
                return self._detect_with_tflite(rgb_frame)
            else:
                return self._detect_with_saved_model(rgb_frame)
        except Exception as e:
            logging.error(f"Error in object detection: {str(e)}")
            return []
    
    def _detect_with_tflite(self, frame):
        """Detect objects using TFLite model"""
        # Get input and output details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        # Prepare input tensor
        input_shape = input_details[0]['shape']
        if input_shape[1:3] != frame.shape[:2]:
            frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
        
        input_tensor = np.expand_dims(frame, axis=0).astype(np.uint8)
        
        # Run inference
        self.interpreter.set_tensor(input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get outputs
        boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(output_details[1]['index'])[0].astype(np.int32)
        scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
        
        return self._process_detections(boxes, classes, scores, frame.shape)
    
    def _detect_with_saved_model(self, frame):
        """Detect objects using SavedModel"""
        # Prepare image for detection
        input_tensor = tf.convert_to_tensor(frame)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        # Run detection
        detections = self.detection_model(input_tensor)
        
        # Process results based on model type
        if isinstance(detections, dict) and 'detection_boxes' in detections:
            # SSD MobileNet format
            boxes = detections['detection_boxes'][0].numpy()
            classes = detections['detection_classes'][0].numpy().astype(np.int32)
            scores = detections['detection_scores'][0].numpy()
            
            return self._process_detections(boxes, classes, scores, frame.shape)
        else:
            # Classification model (fallback) - not implemented here
            return []
    
    def _process_detections(self, boxes, classes, scores, frame_shape):
        """Process detection results into a standard format"""
        # Get detection parameters from env vars or config
        threshold = float(os.getenv('DETECTION_THRESHOLD', 
                                   self.config['detection']['threshold']))
        
        animal_classes = self.config['detection']['animal_classes']
        class_names = self.config['detection']['class_names']
        
        height, width = frame_shape[:2]
        detected_objects = []
        
        # Get max detections from env var or use default
        max_detections = int(os.getenv('MAX_DETECTIONS', 20))
        
        for i in range(min(max_detections, len(scores))):
            if scores[i] > threshold:
                if classes[i] in animal_classes:
                    ymin, xmin, ymax, xmax = boxes[i]
                    
                    # Convert normalized coordinates to pixel coordinates
                    xmin_px = int(xmin * width)
                    ymin_px = int(ymin * height)
                    xmax_px = int(xmax * width)
                    ymax_px = int(ymax * height)
                    
                    detected_objects.append({
                        'class': classes[i],
                        'class_name': class_names.get(classes[i], 'unknown'),
                        'confidence': float(scores[i]),
                        'bbox': (xmin_px, ymin_px, xmax_px, ymax_px),
                        'bbox_norm': (xmin, ymin, xmax, ymax)  # Keep normalized coordinates too
                    })
        
        return detected_objects