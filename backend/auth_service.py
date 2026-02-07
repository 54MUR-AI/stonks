from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import logging
import os
from dotenv import load_dotenv
from backend.models import User
from backend.database import get_db

# Load environment variables
load_dotenv()

# Security Configuration from environment
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY environment variable is not set")

ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate a user by email and password."""
        try:
            user = self.db.query(User).filter(User.email == email).first()
            if not user:
                return None
            if not self.verify_password(password, user.password):
                return None
            return user
        except Exception as e:
            self.logger.error(f"Error authenticating user: {str(e)}")
            raise HTTPException(status_code=500, detail="Authentication failed")

    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new JWT access token."""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({"exp": expire})
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            return encoded_jwt
        except Exception as e:
            self.logger.error(f"Error creating access token: {str(e)}")
            raise HTTPException(status_code=500, detail="Could not create access token")

    def create_user(
        self,
        email: str,
        password: str,
        username: str,
        full_name: Optional[str] = None
    ) -> User:
        """Create a new user."""
        try:
            # Check if user already exists
            if self.db.query(User).filter(User.email == email).first():
                raise HTTPException(status_code=400, detail="Email already registered")
            if self.db.query(User).filter(User.username == username).first():
                raise HTTPException(status_code=400, detail="Username already taken")

            # Create new user
            hashed_password = self.get_password_hash(password)
            user = User(
                email=email,
                username=username,
                full_name=full_name,
                password=hashed_password,
                is_active=True
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error creating user: {str(e)}")
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Could not create user")

    def change_password(
        self,
        user_id: int,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change a user's password."""
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            if not self.verify_password(current_password, user.password):
                raise HTTPException(status_code=400, detail="Incorrect password")
            
            user.password = self.get_password_hash(new_password)
            self.db.commit()
            return True
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error changing password: {str(e)}")
            self.db.rollback()
            raise HTTPException(status_code=500, detail="Could not change password")


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """Get the current user from a JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    try:
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        return user
    except Exception as e:
        logging.error(f"Error getting current user: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error")


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get the current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
