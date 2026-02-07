"""
Database initialization script that runs on startup.
This handles migrations and initial setup without requiring shell access.
"""
import logging
from backend.database import Base, engine
from backend.models import User, Portfolio, Position, Trade, Watchlist, Alert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize database tables on startup"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully!")
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False

if __name__ == "__main__":
    init_database()
