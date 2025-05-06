# socket_events.py
# Handles WebSocket events and client connections - Updated for eventlet

import logging
import time
import base64
import cv2
import datetime
import eventlet
from flask import request
from flask_socketio import emit

class SocketEventHandler:
    def __init__(self, socketio, detection_systems):
        self.socketio = socketio
        self.detection_systems = detection_systems
        self.connected_clients = set()
        self.heatmap_subscribers = set()
        self.video_subscribers = set()
        
        # Rate limiting for frame emission
        self.last_emission_time = {}
        self.emission_interval = 0.033  # ~30fps max
        
        # Register event handlers
        self.register_events()
        
    def register_events(self):
        """Register all socket event handlers"""
        @self.socketio.on('connect')
        def handle_connect():
            client_id = request.sid
            self.connected_clients.add(client_id)
            logging.info(f"Client connected: {client_id}")
            self._update_client_counts()
            
            # Acknowledge connection
            self.socketio.emit('connection_status', {
                'status': 'connected',
                'client_id': client_id,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, to=client_id)
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = request.sid
            self.connected_clients.discard(client_id)
            self.heatmap_subscribers.discard(client_id)
            self.video_subscribers.discard(client_id)
            logging.info(f"Client disconnected: {client_id}")
            self._update_client_counts()
            
        @self.socketio.on('subscribe_video')
        def handle_subscribe_video(data):
            client_id = request.sid
            camera_id = data.get('camera_id', 'cam1')
            self.video_subscribers.add(client_id)
            logging.info(f"Client {client_id} subscribed to video feed for camera {camera_id}")
            self._update_client_counts()
            
            # Send an immediate response to client to confirm subscription
            self.socketio.emit('video_subscription_ack', {
                'status': 'subscribed',
                'camera_id': camera_id,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, to=client_id)
            
        @self.socketio.on('unsubscribe_video')
        def handle_unsubscribe_video():
            client_id = request.sid
            self.video_subscribers.discard(client_id)
            logging.info(f"Client {client_id} unsubscribed from video feed")
            self._update_client_counts()
            
        @self.socketio.on('subscribe_heatmap')
        def handle_subscribe_heatmap(data):
            client_id = request.sid
            camera_id = data.get('camera_id', 'cam1')
            self.heatmap_subscribers.add(client_id)
            logging.info(f"Client {client_id} subscribed to heatmap for camera {camera_id}")
            self._update_client_counts()
            
        @self.socketio.on('unsubscribe_heatmap')
        def handle_unsubscribe_heatmap():
            client_id = request.sid
            self.heatmap_subscribers.discard(client_id)
            logging.info(f"Client {client_id} unsubscribed from heatmap")
            self._update_client_counts()
            
        # Add a debug ping handler to help with testing
        @self.socketio.on('ping_server')
        def handle_ping(data):
            client_id = request.sid
            logging.info(f"Ping received from client {client_id}: {data}")
            self.socketio.emit('pong', {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'message': 'Server is alive',
                'received': data
            }, to=client_id)
    
    def _update_client_counts(self):
        """Update client count for all detection systems"""
        video_count = len(self.video_subscribers)
        heatmap_count = len(self.heatmap_subscribers)
        
        for camera_id, detection_system in self.detection_systems.items():
            detection_system.update_client_count(video_count, heatmap_count)
    
    def emit_frame(self, camera_id, frame):
        """Emit video frame to subscribers"""
        if not self.video_subscribers:
            return
        
        # Rate limiting to avoid overwhelming the network
        current_time = time.time()
        last_time = self.last_emission_time.get(camera_id, 0)
        
        # Skip if we've sent a frame too recently
        if current_time - last_time < self.emission_interval:
            return
        
        # Update the last emission time
        self.last_emission_time[camera_id] = current_time
        
        try:
            # Make sure we actually have a frame
            if frame is None:
                logging.warning(f"Attempted to emit None frame for camera {camera_id}")
                return
                
            # Resize frame before encoding to reduce network load
            small_frame = cv2.resize(frame, (640, 480))
            
            # Optimize JPEG encoding for network
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            _, buffer = cv2.imencode('.jpg', small_frame, encode_params)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Send the frame via WebSocket
            self.socketio.emit('frame_update', {
                'image': jpg_as_text,
                'timestamp': timestamp,
                'camera_id': camera_id
            })
            
            # Log every 300 frames to avoid excessive logging
            if int(current_time) % 30 == 0:
                logging.info(f"Emitted frame for camera {camera_id} to {len(self.video_subscribers)} subscribers")
                
        except Exception as e:
            logging.error(f"Error emitting frame: {str(e)}")
    
    def emit_heatmap(self, camera_id, heatmap_img, points):
        """Emit heatmap data to subscribers"""
        if not self.heatmap_subscribers:
            return
        
        try:
            # Encode heatmap to JPEG
            _, buffer = cv2.imencode('.jpg', heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # Send the heatmap data via WebSocket
            self.socketio.emit('heatmap_update', {
                'image': jpg_as_text,
                'points': points,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'camera_id': camera_id
            })
        except Exception as e:
            logging.error(f"Error emitting heatmap: {str(e)}")
    
    def emit_alert(self, alert_data):
        """Emit alert notification to all connected clients"""
        try:
            self.socketio.emit('new_alert', alert_data)
            logging.info(f"Alert emitted: {alert_data['type']} - {alert_data['details']}")
        except Exception as e:
            logging.error(f"Error emitting alert: {str(e)}")