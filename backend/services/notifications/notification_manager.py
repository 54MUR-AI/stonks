from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import asyncio
import logging
import smtplib
from email.mime.text import MIMEText
import aiohttp
import json

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"

@dataclass
class NotificationConfig:
    type: NotificationType
    enabled: bool = True
    settings: Dict[str, Any] = None

    @classmethod
    def email(cls, smtp_host: str, smtp_port: int, username: str, password: str,
              recipients: List[str], use_tls: bool = True) -> 'NotificationConfig':
        return cls(
            type=NotificationType.EMAIL,
            settings={
                "smtp_host": smtp_host,
                "smtp_port": smtp_port,
                "username": username,
                "password": password,
                "recipients": recipients,
                "use_tls": use_tls
            }
        )

    @classmethod
    def slack(cls, webhook_url: str, channel: str) -> 'NotificationConfig':
        return cls(
            type=NotificationType.SLACK,
            settings={
                "webhook_url": webhook_url,
                "channel": channel
            }
        )

    @classmethod
    def webhook(cls, url: str, headers: Optional[Dict[str, str]] = None) -> 'NotificationConfig':
        return cls(
            type=NotificationType.WEBHOOK,
            settings={
                "url": url,
                "headers": headers or {}
            }
        )

class NotificationSender(ABC):
    @abstractmethod
    async def send(self, subject: str, message: str, metadata: Optional[Dict] = None) -> bool:
        pass

class EmailSender(NotificationSender):
    def __init__(self, config: NotificationConfig):
        self.config = config.settings

    async def send(self, subject: str, message: str, metadata: Optional[Dict] = None) -> bool:
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.config['username']
            msg['To'] = ', '.join(self.config['recipients'])

            with smtplib.SMTP(self.config['smtp_host'], self.config['smtp_port']) as server:
                if self.config['use_tls']:
                    server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

class SlackSender(NotificationSender):
    def __init__(self, config: NotificationConfig):
        self.config = config.settings

    async def send(self, subject: str, message: str, metadata: Optional[Dict] = None) -> bool:
        try:
            payload = {
                "channel": self.config['channel'],
                "text": f"*{subject}*\n{message}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": subject}
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": message}
                    }
                ]
            }

            if metadata:
                payload["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"```{json.dumps(metadata, indent=2)}```"
                    }
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(self.config['webhook_url'], json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

class WebhookSender(NotificationSender):
    def __init__(self, config: NotificationConfig):
        self.config = config.settings

    async def send(self, subject: str, message: str, metadata: Optional[Dict] = None) -> bool:
        try:
            payload = {
                "subject": subject,
                "message": message,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['url'],
                    json=payload,
                    headers=self.config['headers']
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

class NotificationManager:
    def __init__(self):
        self.configs: Dict[str, NotificationConfig] = {}
        self.senders: Dict[str, NotificationSender] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def add_config(self, name: str, config: NotificationConfig):
        """Add a new notification configuration"""
        self.configs[name] = config
        if config.type == NotificationType.EMAIL:
            self.senders[name] = EmailSender(config)
        elif config.type == NotificationType.SLACK:
            self.senders[name] = SlackSender(config)
        elif config.type == NotificationType.WEBHOOK:
            self.senders[name] = WebhookSender(config)

    def remove_config(self, name: str):
        """Remove a notification configuration"""
        self.configs.pop(name, None)
        self.senders.pop(name, None)

    async def start(self):
        """Start the notification worker"""
        if not self._worker_task or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("Notification worker started")

    async def stop(self):
        """Stop the notification worker"""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("Notification worker stopped")

    async def notify(self, subject: str, message: str,
                    targets: Optional[List[str]] = None,
                    metadata: Optional[Dict] = None):
        """Queue a notification for delivery"""
        await self._queue.put({
            "subject": subject,
            "message": message,
            "targets": targets,
            "metadata": metadata,
            "timestamp": datetime.now()
        })

    async def _worker(self):
        """Process queued notifications"""
        while True:
            try:
                notification = await self._queue.get()
                targets = notification.get("targets") or list(self.senders.keys())
                
                for target in targets:
                    if target in self.senders and self.configs[target].enabled:
                        sender = self.senders[target]
                        success = await sender.send(
                            notification["subject"],
                            notification["message"],
                            notification["metadata"]
                        )
                        if success:
                            logger.info(f"Notification sent successfully to {target}")
                        else:
                            logger.error(f"Failed to send notification to {target}")
                
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                await asyncio.sleep(1)  # Back off on error
