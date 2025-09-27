import psycopg2
from psycopg2.extras import RealDictCursor
import json
from typing import List, Dict, Any
from config.settings import settings
import logging

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.setup_database()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD
            )
            return self.connection
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            raise
    
    def setup_database(self):
        """Initialize database tables and pgvector extension"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create contracts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contracts (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_risk_score FLOAT,
                    risk_level VARCHAR(10),
                    processed BOOLEAN DEFAULT FALSE
                );
            """)
            
            # Create clauses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clauses (
                    id SERIAL PRIMARY KEY,
                    contract_id INTEGER REFERENCES contracts(id),
                    clause_text TEXT NOT NULL,
                    clause_type VARCHAR(100),
                    risk_score FLOAT,
                    risk_level VARCHAR(10),
                    compliance_status VARCHAR(20),
                    embedding vector(384)
                );
            """)
            
            # Create risk_assessments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_assessments (
                    id SERIAL PRIMARY KEY,
                    contract_id INTEGER REFERENCES contracts(id),
                    assessment_type VARCHAR(50),
                    assessment_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create suggestions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS suggestions (
                    id SERIAL PRIMARY KEY,
                    clause_id INTEGER REFERENCES clauses(id),
                    suggestion_text TEXT,
                    priority VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            logging.info("Database setup completed successfully")
            
        except Exception as e:
            logging.error(f"Database setup error: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_contract(self, filename: str, content: str) -> int:
        """Insert new contract and return contract ID"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO contracts (filename, content) VALUES (%s, %s) RETURNING id",
                (filename, content)
            )
            contract_id = cursor.fetchone()[0]
            conn.commit()
            return contract_id
        except Exception as e:
            logging.error(f"Error inserting contract: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def insert_clause(self, contract_id: int, clause_text: str, clause_type: str, 
                     risk_score: float, risk_level: str, compliance_status: str, 
                     embedding: List[float]):
        """Insert clause with embedding"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO clauses 
                (contract_id, clause_text, clause_type, risk_score, risk_level, 
                 compliance_status, embedding) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (contract_id, clause_text, clause_type, risk_score, 
                  risk_level, compliance_status, embedding))
            conn.commit()
        except Exception as e:
            logging.error(f"Error inserting clause: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def update_contract_risk_score(self, contract_id: int, risk_score: float, risk_level: str):
        """Update contract's overall risk score"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE contracts 
                SET overall_risk_score = %s, risk_level = %s, processed = TRUE 
                WHERE id = %s
            """, (risk_score, risk_level, contract_id))
            conn.commit()
        except Exception as e:
            logging.error(f"Error updating contract risk score: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()
    
    def search_similar_clauses(self, query_embedding: List[float], limit: int = 5):
        """Search for similar clauses using vector similarity"""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            cursor.execute("""
                SELECT c.clause_text, c.clause_type, c.risk_score, c.risk_level,
                       (c.embedding <-> %s::vector) as distance
                FROM clauses c
                ORDER BY c.embedding <-> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            
            results = cursor.fetchall()
            return [dict(row) for row in results]
        except Exception as e:
            logging.error(f"Error searching similar clauses: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_contract_details(self, contract_id: int):
        """Get contract with all clauses and assessments"""
        conn = self.connect()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get contract info
            cursor.execute("SELECT * FROM contracts WHERE id = %s", (contract_id,))
            contract = dict(cursor.fetchone())
            
            # Get clauses
            cursor.execute("SELECT * FROM clauses WHERE contract_id = %s", (contract_id,))
            clauses = [dict(row) for row in cursor.fetchall()]
            
            # Get assessments
            cursor.execute("SELECT * FROM risk_assessments WHERE contract_id = %s", (contract_id,))
            assessments = [dict(row) for row in cursor.fetchall()]
            
            return {
                "contract": contract,
                "clauses": clauses,
                "assessments": assessments
            }
        except Exception as e:
            logging.error(f"Error getting contract details: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

db_manager = DatabaseManager()
