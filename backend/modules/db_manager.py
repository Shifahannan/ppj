# db_manager.py
import sqlite3
import threading
import logging
import datetime
import base64

class DatabaseManager:
    def __init__(self, db_path, pool_size=5):
        """Initialize the database manager with connection pooling"""
        self.db_path = db_path
        self.pool_size = max(1, pool_size)
        self.connections = []
        self.connection_index = 0
        self.lock = threading.Lock()
        
        # Initialize connection pool
        self._initialize_pool()
        
        # Create required tables and indexes
        self._initialize_schema()
        
        logging.info(f"Database manager initialized with connection pool of size {pool_size}")
    
    def _initialize_pool(self):
        """Create a pool of database connections"""
        try:
            for _ in range(self.pool_size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row  # Return rows as dictionaries
                self.connections.append(conn)
        except sqlite3.Error as e:
            logging.error(f"Error creating connection pool: {str(e)}")
            # If pool creation fails, we'll create connections as needed
    
    def _initialize_schema(self):
        """Create necessary tables and indexes if they don't exist"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                alert_type TEXT,
                details TEXT,
                confidence REAL,
                image BLOB,
                processed INTEGER DEFAULT 0,
                camera_id TEXT
            )
            ''')
            
            # Create heatmap data table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS heatmap_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                x INTEGER,
                y INTEGER,
                intensity REAL,
                camera_id TEXT
            )
            ''')
            
            # Add indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_heatmap_timestamp ON heatmap_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_heatmap_camera ON heatmap_data(camera_id)')
            
            conn.commit()
            logging.info("Database schema initialized successfully")
        except sqlite3.Error as e:
            logging.error(f"Database schema initialization error: {str(e)}")
    
    def get_connection(self):
        """Get a connection from the pool or create a new one if the pool is empty"""
        if not self.connections:
            try:
                # Create a new connection if pool is empty
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                return conn
            except sqlite3.Error as e:
                logging.error(f"Error creating database connection: {str(e)}")
                raise
        
        # Get a connection from the pool
        with self.lock:
            conn = self.connections[self.connection_index]
            self.connection_index = (self.connection_index + 1) % len(self.connections)
            return conn
    
    def execute_query(self, query, params=None):
        """Execute a query and return the results"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Query execution error: {query} - {str(e)}")
            conn.rollback()
            raise
    
    def execute_update(self, query, params=None):
        """Execute an update query and return affected row count"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            logging.error(f"Update execution error: {query} - {str(e)}")
            conn.rollback()
            raise
    
    def store_alert(self, alert_type, image_data, details, confidence, camera_id):
        """Store an alert with an image in the database"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query = """
                INSERT INTO alerts (timestamp, alert_type, details, confidence, image, camera_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (timestamp, alert_type, details, confidence, image_data, camera_id))
            alert_id = cursor.lastrowid
            conn.commit()
            
            return {
                'id': alert_id,
                'timestamp': timestamp,
                'type': alert_type,
                'details': details,
                'confidence': confidence,
                'camera_id': camera_id,
                'image': base64.b64encode(image_data).decode('utf-8')
            }
        except sqlite3.Error as e:
            logging.error(f"Error storing alert: {str(e)}")
            raise
    
    def store_heatmap_point(self, x, y, intensity, camera_id):
        """Store a heatmap data point in the database"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query = """
                INSERT INTO heatmap_data (timestamp, x, y, intensity, camera_id)
                VALUES (?, ?, ?, ?, ?)
            """
            self.execute_update(query, (timestamp, x, y, intensity, camera_id))
        except sqlite3.Error as e:
            logging.error(f"Error storing heatmap point: {str(e)}")
    
    def get_alerts(self, page=1, per_page=10, alert_type=None, camera_id=None):
        """Get paginated alerts with optional filtering"""
        try:
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
            
            # Add pagination
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([per_page, (page - 1) * per_page])
            
            # Execute query
            alerts = self.execute_query(query, params)
            
            # Get total count
            count_query = "SELECT COUNT(*) as count FROM alerts"
            if where_clauses:
                count_query += " WHERE " + " AND ".join(where_clauses)
                count_result = self.execute_query(count_query, params[:-2])
            else:
                count_result = self.execute_query(count_query)
            
            total = dict(count_result[0])['count'] if count_result else 0
            
            return {
                'alerts': [dict(row) for row in alerts],
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total': total,
                    'pages': (total + per_page - 1) // per_page
                }
            }
        except sqlite3.Error as e:
            logging.error(f"Error getting alerts: {str(e)}")
            raise
    
    def get_alert_by_id(self, alert_id):
        """Get a specific alert with its image"""
        try:
            query = """
                SELECT id, timestamp, alert_type, details, confidence, image, camera_id 
                FROM alerts WHERE id = ?
            """
            results = self.execute_query(query, [alert_id])
            
            if not results:
                return None
            
            alert = dict(results[0])
            # Convert binary image to base64
            if alert['image']:
                alert['image'] = base64.b64encode(alert['image']).decode('utf-8')
            
            return alert
        except sqlite3.Error as e:
            logging.error(f"Error getting alert by ID: {str(e)}")
            raise
    
    def get_heatmap_points(self, hours=24, camera_id=None):
        """Get heatmap data from the last N hours"""
        try:
            query = """
                SELECT x, y, intensity FROM heatmap_data 
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
            """
            params = [hours]
            
            if camera_id:
                query += " AND camera_id = ?"
                params.append(camera_id)
            
            results = self.execute_query(query, params)
            return [dict(row) for row in results]
        except sqlite3.Error as e:
            logging.error(f"Error getting heatmap data: {str(e)}")
            raise
    
    def get_alert_stats(self):
        """Get detection statistics"""
        try:
            stats = {}
            
            # Count by type in the last 24 hours
            type_query = """
                SELECT alert_type, COUNT(*) as count FROM alerts 
                WHERE timestamp >= datetime('now', '-24 hours') 
                GROUP BY alert_type
            """
            type_results = self.execute_query(type_query)
            stats['by_type'] = {row['alert_type']: row['count'] for row in type_results}
            
            # Count by hour of day
            hour_query = """
                SELECT strftime('%H', timestamp) as hour, COUNT(*) as count 
                FROM alerts 
                WHERE timestamp >= datetime('now', '-7 days') 
                GROUP BY hour 
                ORDER BY hour
            """
            hour_results = self.execute_query(hour_query)
            stats['by_hour'] = {row['hour']: row['count'] for row in hour_results}
            
            # Average confidence by type
            confidence_query = """
                SELECT alert_type, AVG(confidence) as avg_confidence 
                FROM alerts 
                GROUP BY alert_type
            """
            confidence_results = self.execute_query(confidence_query)
            stats['avg_confidence'] = {row['alert_type']: row['avg_confidence'] for row in confidence_results}
            
            return stats
        except sqlite3.Error as e:
            logging.error(f"Error getting alert statistics: {str(e)}")
            raise
    
    def cleanup(self):
        """Close all database connections in the pool"""
        if self.connections:
            for conn in self.connections:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass  # Ignore errors during cleanup
        
        self.connections = []
        logging.info("Database connections closed")
