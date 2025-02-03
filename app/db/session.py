import os
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db.models import Base  # Ensure you import your models here

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/ml_registry.db")

# Ensure SQLite database directory and file exist
if DATABASE_URL.startswith("sqlite:///"):
    # Extract the database file path
    db_path = DATABASE_URL.replace("sqlite:///", "")
    dirpath = os.path.dirname(db_path)

    # Create the directory if it does not exist
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    # Create the SQLite database file if it does not exist
    if not os.path.exists(db_path):
        open(db_path, "w").close()  # Create an empty file
        try:
            conn = sqlite3.connect(db_path)
            conn.close()
        except Exception as e:
            print(f"Error creating SQLite database file: {e}")

# Create the SQLAlchemy engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# **Ensure database schema is created**
def init_db():
    """Creates tables if they do not exist"""
    Base.metadata.create_all(bind=engine)


init_db()


def get_db():
    """
    Dependency function for FastAPI that provides a SQLAlchemy session.
    It yields a session and ensures it is closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
