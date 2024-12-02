from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from models import Activity, ActivityType, Notification, NotificationType, User
from email_service import fastmail
from fastapi_mail import MessageSchema
import json
from datetime import datetime

class ActivityService:
    @staticmethod
    async def create_activity(
        db: Session,
        user_id: int,
        activity_type: ActivityType,
        target_id: int,
        target_type: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Activity:
        """Create a new activity entry"""
        activity = Activity(
            user_id=user_id,
            activity_type=activity_type,
            target_id=target_id,
            target_type=target_type,
            data=data or {}
        )
        db.add(activity)
        db.commit()
        db.refresh(activity)
        return activity
    
    @staticmethod
    def get_user_activities(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> list[Activity]:
        """Get activities for a specific user"""
        return db.query(Activity).filter(
            Activity.user_id == user_id
        ).order_by(Activity.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_feed_activities(
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> list[Activity]:
        """Get activities for user's feed (including followed users)"""
        # Get IDs of users being followed
        following_ids = [follow.following_id for follow in db.query(User).get(user_id).following]
        
        # Get activities from user and followed users
        return db.query(Activity).filter(
            Activity.user_id.in_([user_id] + following_ids)
        ).order_by(Activity.created_at.desc()).offset(skip).limit(limit).all()

class NotificationService:
    @staticmethod
    async def create_notification(
        db: Session,
        user_id: int,
        notification_type: NotificationType,
        data: Dict[str, Any]
    ) -> Notification:
        """Create a new notification"""
        notification = Notification(
            user_id=user_id,
            notification_type=notification_type,
            data=data
        )
        db.add(notification)
        db.commit()
        db.refresh(notification)
        
        # Get user's notification preferences
        user = db.query(User).get(user_id)
        
        # Send email notification if enabled
        if user.notification_preferences.get("email", True):
            await NotificationService.send_email_notification(user.email, notification)
        
        return notification
    
    @staticmethod
    async def send_email_notification(email: str, notification: Notification):
        """Send email notification"""
        subject = NotificationService.get_notification_subject(notification)
        body = NotificationService.get_notification_body(notification)
        
        message = MessageSchema(
            subject=subject,
            recipients=[email],
            body=body,
            subtype="html"
        )
        
        await fastmail.send_message(message)
    
    @staticmethod
    def get_notification_subject(notification: Notification) -> str:
        """Generate notification email subject"""
        subjects = {
            NotificationType.PORTFOLIO_SHARED: "Portfolio Shared with You",
            NotificationType.NEW_FOLLOWER: "New Follower",
            NotificationType.NEW_COMMENT: "New Comment on Your Portfolio",
            NotificationType.PRICE_ALERT: "Price Alert Triggered",
            NotificationType.PORTFOLIO_MENTION: "You Were Mentioned in a Comment",
            NotificationType.TRADE_ALERT: "Trade Alert"
        }
        return subjects.get(notification.notification_type, "Stonks Notification")
    
    @staticmethod
    def get_notification_body(notification: Notification) -> str:
        """Generate notification email body"""
        template = """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>{title}</h2>
            <p>{message}</p>
            <p style="color: #666; font-size: 0.9em;">
                Sent at: {timestamp}
            </p>
            <hr>
            <p style="font-size: 0.8em; color: #999;">
                You received this notification because you have email notifications enabled.
                To update your notification preferences, visit your account settings.
            </p>
        </div>
        """
        
        title = NotificationService.get_notification_subject(notification)
        message = NotificationService.get_notification_message(notification)
        timestamp = notification.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return template.format(title=title, message=message, timestamp=timestamp)
    
    @staticmethod
    def get_notification_message(notification: Notification) -> str:
        """Generate notification message based on type and data"""
        data = notification.data
        
        messages = {
            NotificationType.PORTFOLIO_SHARED: (
                f"User {data.get('shared_by')} has shared their portfolio "
                f"'{data.get('portfolio_name')}' with you."
            ),
            NotificationType.NEW_FOLLOWER: (
                f"User {data.get('follower')} is now following you."
            ),
            NotificationType.NEW_COMMENT: (
                f"User {data.get('commenter')} commented on your portfolio "
                f"'{data.get('portfolio_name')}': {data.get('comment')}"
            ),
            NotificationType.PRICE_ALERT: (
                f"Price alert triggered for {data.get('symbol')}: "
                f"Current price: ${data.get('current_price'):.2f}"
            ),
            NotificationType.PORTFOLIO_MENTION: (
                f"User {data.get('mentioner')} mentioned you in a comment: "
                f"{data.get('comment')}"
            ),
            NotificationType.TRADE_ALERT: (
                f"Trade alert for {data.get('symbol')}: {data.get('message')}"
            )
        }
        
        return messages.get(
            notification.notification_type,
            "You have a new notification."
        )
    
    @staticmethod
    def get_user_notifications(
        db: Session,
        user_id: int,
        unread_only: bool = False,
        skip: int = 0,
        limit: int = 50
    ) -> list[Notification]:
        """Get notifications for a user"""
        query = db.query(Notification).filter(Notification.user_id == user_id)
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        return query.order_by(Notification.created_at.desc()).offset(skip).limit(limit).all()
    
    @staticmethod
    def mark_as_read(db: Session, notification_id: int, user_id: int) -> bool:
        """Mark a notification as read"""
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()
        
        if notification:
            notification.is_read = True
            db.commit()
            return True
        return False
    
    @staticmethod
    def mark_all_as_read(db: Session, user_id: int) -> int:
        """Mark all notifications as read for a user"""
        result = db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_read == False
        ).update({"is_read": True})
        
        db.commit()
        return result
