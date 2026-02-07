"""
Admin API endpoints for database management without shell access.
These endpoints allow you to run migrations, seed data, and perform admin tasks via HTTP.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.database import SessionLocal, Base, engine
from backend.models import User
from backend.auth_service import get_current_active_user
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/migrate")
async def run_migrations():
    """
    Run database migrations (create all tables).
    This is a workaround for no shell access on Render free tier.
    """
    try:
        Base.metadata.create_all(bind=engine)
        return {
            "status": "success",
            "message": "Database migrations completed successfully"
        }
    except Exception as e:
        logger.error(f"Migration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/seed-demo-data")
async def seed_demo_data(db: Session = Depends(get_db)):
    """
    Seed the database with demo data for testing.
    """
    try:
        # Check if demo user already exists
        demo_user = db.query(User).filter(User.email == "demo@stonks.com").first()
        if demo_user:
            return {
                "status": "info",
                "message": "Demo data already exists"
            }
        
        # Create demo user
        from backend.auth_service import AuthService
        auth_service = AuthService()
        
        demo_user = User(
            username="demo",
            email="demo@stonks.com",
            hashed_password=auth_service.get_password_hash("demo123"),
            is_active=True
        )
        db.add(demo_user)
        db.commit()
        
        return {
            "status": "success",
            "message": "Demo data seeded successfully",
            "credentials": {
                "email": "demo@stonks.com",
                "password": "demo123"
            }
        }
    except Exception as e:
        logger.error(f"Seed error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/db-status")
async def check_database_status(db: Session = Depends(get_db)):
    """
    Check database connection and table status.
    """
    try:
        # Try to query users table
        user_count = db.query(User).count()
        
        return {
            "status": "connected",
            "tables_exist": True,
            "user_count": user_count,
            "database_url": "***hidden***"
        }
    except Exception as e:
        return {
            "status": "error",
            "tables_exist": False,
            "error": str(e)
        }

@router.post("/reset-database")
async def reset_database():
    """
    ⚠️ DANGER: Drop all tables and recreate them.
    Use with caution - this will delete all data!
    """
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        return {
            "status": "success",
            "message": "Database reset successfully - all data deleted"
        }
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
