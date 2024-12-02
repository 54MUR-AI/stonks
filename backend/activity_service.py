from sqlalchemy.orm import Session
from sqlalchemy import desc
from fastapi import HTTPException
import logging
from typing import List, Dict, Any

from models import Activity, Notification, User, ActivityType, NotificationType


class ActivityService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def create_activity(
        self,
        user_id: int,
        activity_type: str,
        target_id: int,
        target_type: str,
        data: Dict[str, Any] = None
    ) -> Activity:
        try:
            activity = Activity(
                user_id=user_id,
                activity_type=activity_type,
                target_id=target_id,
                target_type=target_type,
                data=data or {}
            )
            self.db.add(activity)
            self.db.commit()
            self.db.refresh(activity)
            return activity
        except Exception as e:
            self.logger.error(f"Error creating activity: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create activity")

    def get_user_activities(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 50
    ) -> List[Activity]:
        try:
            return (
                self.db.query(Activity)
                .filter(Activity.user_id == user_id)
                .order_by(desc(Activity.created_at))
                .offset(skip)
                .limit(limit)
                .all()
            )
        except Exception as e:
            self.logger.error(f"Error fetching user activities: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch activities")


class NotificationService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def create_notification(
        self,
        user_id: int,
        notification_type: str,
        data: Dict[str, Any] = None
    ) -> Notification:
        try:
            notification = Notification(
                user_id=user_id,
                notification_type=notification_type,
                data=data or {},
                is_read=False
            )
            self.db.add(notification)
            self.db.commit()
            self.db.refresh(notification)
            return notification
        except Exception as e:
            self.logger.error(f"Error creating notification: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create notification")

    def get_user_notifications(
        self,
        user_id: int,
        unread_only: bool = False,
        skip: int = 0,
        limit: int = 50
    ) -> List[Notification]:
        try:
            query = self.db.query(Notification).filter(Notification.user_id == user_id)
            if unread_only:
                query = query.filter(Notification.is_read.is_(False))
            return query.order_by(desc(Notification.created_at)).offset(skip).limit(limit).all()
        except Exception as e:
            self.logger.error(f"Error fetching user notifications: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch notifications")

    def mark_notification_read(self, notification_id: int, user_id: int) -> Notification:
        try:
            notification = (
                self.db.query(Notification)
                .filter(Notification.id == notification_id, Notification.user_id == user_id)
                .first()
            )
            if not notification:
                raise HTTPException(status_code=404, detail="Notification not found")

            notification.is_read = True
            self.db.commit()
            self.db.refresh(notification)
            return notification
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error marking notification as read: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update notification")

    def mark_all_notifications_read(self, user_id: int) -> None:
        try:
            unread_notifications = (
                self.db.query(Notification)
                .filter(Notification.user_id == user_id, Notification.is_read.is_(False))
                .all()
            )
            for notification in unread_notifications:
                notification.is_read = True
            self.db.commit()
        except Exception as e:
            self.logger.error(f"Error marking all notifications as read: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to update notifications")
