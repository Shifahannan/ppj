# routes.py
# Flask API routes for the stray animal and flood detection system

import logging
import sqlite3
import base64
from flask import jsonify, request, send_from_directory, Blueprint

# Create a Blueprint for our API routes
api_blueprint = Blueprint('api', __name__)

def get_db_path(camera_id, detection_systems):
    """Determine which database to use based on camera ID"""
    if camera_id and camera_id in detection_systems:
        return detection_systems[camera_id].db_path
    # Default to the first camera's database or main database
    db_path = 'detections.db'
    if detection_systems:
        db_path = next(iter(detection_systems.values())).db_path
    return db_path

@api_blueprint.route('/alerts', methods=['GET'])
def get_alerts(detection_systems):
    """Get alerts from database with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        alert_type = request.args.get('type', None)
        camera_id = request.args.get('camera_id', None)
        
        # Get the appropriate database path
        db_path = get_db_path(camera_id, detection_systems)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query based on filters
        query = "SELECT id, timestamp, alert_type, details, confidence, camera_id FROM alerts"
        params = []
        where_clauses = []
        
        if alert_type:
            where_clauses.append("alert_type = ?")
            params.append(alert_type)
        
        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([per_page, (page - 1) * per_page])
        
        # Execute query
        cursor.execute(query, params)
        alerts = [dict(row) for row in cursor.fetchall()]
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM alerts"
        if where_clauses:
            count_query += " WHERE " + " AND ".join(where_clauses)
            cursor.execute(count_query, params[:-2])
        else:
            cursor.execute(count_query)
        
        total = cursor.fetchone()[0]
        conn.close()
        
        return jsonify({
            'alerts': alerts,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total,
                'pages': (total + per_page - 1) // per_page
            }
        })
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_blueprint.route('/alerts/<int:alert_id>', methods=['GET'])
def get_alert(alert_id, detection_systems):
    """Get a specific alert with image"""
    try:
        # Use the first available database
        db_path = get_db_path(None, detection_systems)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, timestamp, alert_type, details, confidence, image, camera_id FROM alerts WHERE id = ?",
            [alert_id]
        )
        
        alert = cursor.fetchone()
        conn.close()
        
        if alert:
            alert_dict = dict(alert)
            # Convert binary image to base64
            alert_dict['image'] = base64.b64encode(alert_dict['image']).decode('utf-8')
            return jsonify(alert_dict)
        else:
            return jsonify({'error': 'Alert not found'}), 404
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_blueprint.route('/heatmap', methods=['GET'])
def get_heatmap_data(detection_systems):
    """Get historical heatmap data with time filtering"""
    try:
        hours = request.args.get('hours', 24, type=int)
        camera_id = request.args.get('camera_id', None)
        
        # Get the appropriate database path
        db_path = get_db_path(camera_id, detection_systems)
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query with camera filtering if specified
        query = "SELECT x, y, intensity FROM heatmap_data WHERE timestamp >= datetime('now', '-' || ? || ' hours')"
        params = [hours]
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        # Get heatmap data from the last N hours
        cursor.execute(query, params)
        
        points = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({'points': points})
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_blueprint.route('/stats', methods=['GET'])
def get_stats(detection_systems):
    """Get detection statistics"""
    try:
        camera_id = request.args.get('camera_id', None)
        db_path = get_db_path(camera_id, detection_systems)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build base query with camera filtering if needed
        base_query = "FROM alerts"
        params = []
        
        if camera_id:
            base_query += " WHERE camera_id = ?"
            params.append(camera_id)
        
        # Count by type in the last 24 hours
        type_query = f"SELECT alert_type, COUNT(*) as count {base_query} AND timestamp >= datetime('now', '-24 hours') GROUP BY alert_type"
        if camera_id:
            cursor.execute(type_query, params)
        else:
            cursor.execute(type_query.replace("AND timestamp", "WHERE timestamp"))
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by hour of day
        hour_query = f"SELECT strftime('%H', timestamp) as hour, COUNT(*) as count {base_query} AND timestamp >= datetime('now', '-7 days') GROUP BY hour ORDER BY hour"
        if camera_id:
            cursor.execute(hour_query, params)
        else:
            cursor.execute(hour_query.replace("AND timestamp", "WHERE timestamp"))
        hourly_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average confidence by type
        avg_query = f"SELECT alert_type, AVG(confidence) as avg_confidence {base_query} GROUP BY alert_type"
        cursor.execute(avg_query, params if camera_id else [])
        avg_confidence = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return jsonify({
            'by_type': type_counts,
            'by_hour': hourly_counts,
            'avg_confidence': avg_confidence
        })
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_blueprint.route('/cameras', methods=['GET'])
def get_cameras(detection_systems):
    """Get list of available cameras"""
    try:
        cameras = []
        for camera_id, system in detection_systems.items():
            cameras.append({
                'id': camera_id,
                'url': system.camera_url
            })
        
        return jsonify({'cameras': cameras})
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def register_routes(app, detection_systems):
    """Register all routes with the app"""
    # Create a wrapper for each route function that injects detection_systems
    def wrap_route(route_func):
        def wrapped(*args, **kwargs):
            return route_func(detection_systems=detection_systems, *args, **kwargs)
        wrapped.__name__ = route_func.__name__
        return wrapped
    
    # Register each API endpoint with the wrapper
    app.add_url_rule('/api/alerts', 'get_alerts', wrap_route(get_alerts))
    app.add_url_rule('/api/alerts/<int:alert_id>', 'get_alert', wrap_route(get_alert))
    app.add_url_rule('/api/heatmap', 'get_heatmap_data', wrap_route(get_heatmap_data))
    app.add_url_rule('/api/stats', 'get_stats', wrap_route(get_stats))
    app.add_url_rule('/api/cameras', 'get_cameras', wrap_route(get_cameras))