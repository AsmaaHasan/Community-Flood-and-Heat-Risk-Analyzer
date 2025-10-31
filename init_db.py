"""
Database initialization script
Run this to create all tables in the PostgreSQL database
"""

from database import init_database, get_database_engine
from sqlalchemy import inspect

def main():
    print("Initializing database...")
    
    try:
        # Initialize database and create tables
        engine = init_database()
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"\n✅ Database initialized successfully!")
        print(f"\n📊 Created {len(tables)} tables:")
        for table in sorted(tables):
            print(f"  - {table}")
        
        # Show table details
        print("\n📋 Table Details:")
        for table_name in sorted(tables):
            columns = inspector.get_columns(table_name)
            print(f"\n  {table_name}:")
            for col in columns:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"    - {col['name']}: {col['type']} {nullable}")
        
        print("\n✨ Database is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Error initializing database: {e}")
        raise

if __name__ == "__main__":
    main()
