import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import hashlib
import secrets
import os
from datetime import datetime
from typing import Optional, Dict, List

class UserAuth:
    def __init__(self, db_config: Dict = None):
        """
        Initialize user authentication system
        
        Args:
            db_config (dict): PostgreSQL connection configuration
        """
        self.db_config = db_config or {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'nutrition_rag_users'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'port': os.getenv('DB_PORT', '5432')
        }
        self._ensure_database_exists()
        self.init_database()

    def _ensure_database_exists(self):
        """Create the application database if it does not exist yet."""
        db_name = self.db_config["database"]
        maintenance_config = {**self.db_config, "database": "postgres"}
        try:
            conn = psycopg2.connect(**maintenance_config)
        except psycopg2.OperationalError:
            maintenance_config["database"] = "template1"
            conn = psycopg2.connect(**maintenance_config)

        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (db_name,),
            )
            if not cursor.fetchone():
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                print(f"Created database '{db_name}'")
            cursor.close()
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    salt VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Create user_documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_documents (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    document_id VARCHAR(255) NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    doc_type VARCHAR(50) DEFAULT 'clinical',
                    access_level VARCHAR(20) DEFAULT 'private',
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Database initialization error: {e}")
            raise
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000
        ).hex()
        
        return password_hash, salt
    
    def register_user(self, user_id: str, password: str) -> bool:
        """Register a new user"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            if cursor.fetchone():
                print(f"User {user_id} already exists")
                return False
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Insert new user
            cursor.execute("""
                INSERT INTO users (user_id, password_hash, salt)
                VALUES (%s, %s, %s)
            """, (user_id, password_hash, salt))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"User {user_id} registered successfully")
            return True
            
        except Exception as e:
            print(f"Registration error: {e}")
            return False
    
    def authenticate_user(self, user_id: str, password: str) -> bool:
        """Authenticate user login"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Get user's salt and hash
            cursor.execute("""
                SELECT password_hash, salt FROM users 
                WHERE user_id = %s AND is_active = TRUE
            """, (user_id,))
            
            result = cursor.fetchone()
            if not result:
                print(f"User {user_id} not found or inactive")
                return False
            
            stored_hash, salt = result
            
            # Verify password
            password_hash, _ = self.hash_password(password, salt)
            
            if password_hash == stored_hash:
                # Update last login
                cursor.execute("""
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE user_id = %s
                """, (user_id,))
                
                conn.commit()
                cursor.close()
                conn.close()
                
                print(f"User {user_id} authenticated successfully")
                return True
            else:
                print(f"Invalid password for user {user_id}")
                return False
                
        except Exception as e:
            print(f"Authentication error: {e}")
            return False
    
    def add_user_document(self, user_id: str, document_id: str, filename: str, 
                         doc_type: str = 'clinical', access_level: str = 'private') -> bool:
        """Add document reference for user"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO user_documents (user_id, document_id, filename, doc_type, access_level)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, document_id, filename, doc_type, access_level))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"Document {filename} added for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def get_user_documents(self, user_id: str) -> List[Dict]:
        """Get all documents for a user"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT document_id, filename, doc_type, access_level, added_at
                FROM user_documents 
                WHERE user_id = %s
                ORDER BY added_at DESC
            """, (user_id,))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'document_id': row[0],
                    'filename': row[1],
                    'doc_type': row[2],
                    'access_level': row[3],
                    'added_at': row[4]
                })
            
            cursor.close()
            conn.close()
            
            return documents
            
        except Exception as e:
            print(f"Error getting user documents: {e}")
            return []
    
    def remove_user_document(self, user_id: str, document_id: str) -> bool:
        """Remove document reference for user"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM user_documents 
                WHERE user_id = %s AND document_id = %s
            """, (user_id, document_id))
            
            if cursor.rowcount == 0:
                print(f"Document {document_id} not found for user {user_id}")
                return False
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"Document {document_id} removed for user {user_id}")
            return True
            
        except Exception as e:
            print(f"Error removing document: {e}")
            return False
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            return result is not None
            
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return False 