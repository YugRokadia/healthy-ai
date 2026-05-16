#!/usr/bin/env python3
"""
Vector Database Initialization Script
Sets up LanceDB with the required schema and embedding model
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_db import NutritionHealthVectorDB
from dotenv import load_dotenv

def setup_vector_db():
    """Initialize the vector databases"""
    print("=" * 60)
    print("NutritionRAG - Health Document Vector Database Setup")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        import lancedb
        import pyarrow as pa
        
        dbs = ["./diagnosis_vectordb", "./nutrition_vectordb"]
        embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        schema = pa.schema([
            ("id", pa.int64()),
            ("chunk_id", pa.int64()),
            ("content", pa.string()),
            ("embedding", pa.list_(pa.float32(), embedding_dim)),
            ("source", pa.string()),
            ("chunk_index", pa.int64()),
            ("user_id", pa.string()),
            ("document_id", pa.string()),
            ("upload_timestamp", pa.timestamp('us')),
            ("access_level", pa.string()),
            ("type", pa.string())
        ])
        
        table_name = f"health_nutrition_documents_{embedding_dim}"
        
        for db_path in dbs:
            print(f"\nSetting up database at {db_path}...")
            db = lancedb.connect(db_path)
            print("   ✓ LanceDB connection established")
            
            if table_name not in db.table_names():
                db.create_table(table_name, schema=schema)
                print(f"   ✓ Table '{table_name}' created")
            else:
                print(f"   ✓ Table '{table_name}' already exists")
            
            tables = db.table_names()
            print(f"   ✓ Available tables: {tables}")
            
        print("\n" + "=" * 60)
        print("Vector Database Setup Complete!")
        print("=" * 60)
        print(f"\nDatabase Locations: {', '.join(dbs)}")
        print(f"Default Table: {table_name}")
        print(f"Embedding Model: all-MiniLM-L6-v2 (will auto-load on first use)")
        print(f"Embedding Dimension: {embedding_dim}")
        print("\nReady to process documents!")
        
    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = setup_vector_db()
    sys.exit(0 if success else 1)
