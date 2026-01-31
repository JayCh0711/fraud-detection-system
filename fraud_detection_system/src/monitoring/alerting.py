"""
Alerting System for Fraud Detection Monitoring

Supports:
1. Console Alerts
2. File-based Alerts
3. Email Alerts (Optional)
4. Slack Alerts (Optional)
5. Webhook Alerts (Optional)
"""

import os
import sys
import json
import smtplib
from typing import Dict, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid

from src.monitoring.config import monitoring_config, monitoring_settings
from src.logger import logger
from src.utils.common import write_json, create_directories

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class Alert:
    """Alert data structure"""
    
    alert_id: str
    alert_type: str  # drift, performance, system, business
    severity: str  # info, warning, critical
    title: str
    message: str
    source: str
    
    # Details
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    
    # Status
    status: str = "open"  # open, acknowledged, resolved
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    
    # Timestamps
    created_at: str = None
    updated_at: str = None
    
    # Notification status
    notifications_sent: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.notifications_sent is None:
            self.notifications_sent = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AlertChannel:
    """Base class for alert channels"""
    
    def send(self, alert: Alert) -> bool:
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """Console output for alerts"""
    
    def send(self, alert: Alert) -> bool:
        severity_icons = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨"
        }
        
        icon = severity_icons.get(alert.severity, "📢")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{icon} ALERT: {alert.title}")
        logger.info(f"{'='*60}")
        logger.info(f"Type: {alert.alert_type}")
        logger.info(f"Severity: {alert.severity.upper()}")
        logger.info(f"Source: {alert.source}")
        logger.info(f"Message: {alert.message}")
        
        if alert.metric_name:
            logger.info(f"Metric: {alert.metric_name}")
            logger.info(f"Current Value: {alert.current_value}")
            logger.info(f"Threshold: {alert.threshold_value}")
        
        logger.info(f"Time: {alert.created_at}")
        logger.info(f"{'='*60}\n")
        
        return True


class FileAlertChannel(AlertChannel):
    """File-based alert storage"""
    
    def __init__(self, alerts_dir: str):
        self.alerts_dir = alerts_dir
        create_directories([alerts_dir])
    
    def send(self, alert: Alert) -> bool:
        try:
            alert_path = os.path.join(
                self.alerts_dir,
                f"{alert.alert_id}.json"
            )
            write_json(alert_path, alert.to_dict())
            return True
        except Exception as e:
            logger.error(f"Failed to save alert: {str(e)}")
            return False


class EmailAlertChannel(AlertChannel):
    """Email alert channel"""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_address: str,
        to_addresses: List[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.to_addresses = to_addresses
    
    def send(self, alert: Alert) -> bool:
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.to_addresses)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            body = f"""
Fraud Detection Alert
=====================

Alert ID: {alert.alert_id}
Type: {alert.alert_type}
Severity: {alert.severity}
Source: {alert.source}

Message:
{alert.message}

Metric: {alert.metric_name or 'N/A'}
Current Value: {alert.current_value or 'N/A'}
Threshold: {alert.threshold_value or 'N/A'}

Time: {alert.created_at}

---
This is an automated alert from the Fraud Detection System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False


class SlackAlertChannel(AlertChannel):
    """Slack webhook alert channel"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send(self, alert: Alert) -> bool:
        if not REQUESTS_AVAILABLE:
            logger.warning("Requests library not available for Slack alerts")
            return False
        
        try:
            color_map = {
                "info": "#36a64f",
                "warning": "#ffcc00",
                "critical": "#ff0000"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "#808080"),
                        "title": f"🚨 {alert.title}",
                        "text": alert.message,
                        "fields": [
                            {"title": "Type", "value": alert.alert_type, "short": True},
                            {"title": "Severity", "value": alert.severity.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.created_at, "short": True},
                        ],
                        "footer": "Fraud Detection Monitoring"
                    }
                ]
            }
            
            if alert.metric_name:
                payload["attachments"][0]["fields"].extend([
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Value", "value": str(alert.current_value), "short": True},
                ])
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False


class AlertManager:
    """
    Central alert management system.
    
    Manages alert creation, routing, and notification.
    """
    
    def __init__(self, config: monitoring_config = None):
        """
        Initialize Alert Manager.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or monitoring_config
        self.settings = monitoring_settings
        
        self.channels: List[AlertChannel] = []
        self.alerts: List[Alert] = []
        
        self._setup_channels()
        
        logger.info("Alert Manager initialized")
    
    def _setup_channels(self) -> None:
        """Setup alert channels based on configuration."""
        # Always add console and file channels
        self.channels.append(ConsoleAlertChannel())
        self.channels.append(FileAlertChannel(self.config.alerts_dir))
        
        # Add Slack if configured
        if (self.settings.ENABLE_SLACK_ALERTS and 
            self.settings.SLACK_WEBHOOK_URL):
            self.channels.append(
                SlackAlertChannel(self.settings.SLACK_WEBHOOK_URL)
            )
            logger.info("Slack alerts enabled")
    
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        source: str = "monitoring",
        metric_name: str = None,
        current_value: float = None,
        threshold_value: float = None
    ) -> Alert:
        """
        Create a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            metric_name: Name of metric (optional)
            current_value: Current metric value (optional)
            threshold_value: Threshold value (optional)
        
        Returns:
            Created Alert object
        """
        alert = Alert(
            alert_id=f"ALERT_{uuid.uuid4().hex[:12].upper()}",
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value
        )
        
        self.alerts.append(alert)
        
        return alert
    
    def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """
        Send alert through all configured channels.
        
        Args:
            alert: Alert to send
        
        Returns:
            Dictionary of channel results
        """
        results = {}
        
        for channel in self.channels:
            channel_name = type(channel).__name__
            try:
                success = channel.send(alert)
                results[channel_name] = success
                
                if success:
                    alert.notifications_sent.append(channel_name)
                    
            except Exception as e:
                logger.error(f"Channel {channel_name} failed: {str(e)}")
                results[channel_name] = False
        
        alert.updated_at = datetime.now().isoformat()
        
        return results
    
    def create_and_send(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        **kwargs
    ) -> Tuple[Alert, Dict[str, bool]]:
        """
        Create and send an alert in one call.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            **kwargs: Additional alert parameters
        
        Returns:
            Tuple of (Alert, send results)
        """
        alert = self.create_alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            **kwargs
        )
        
        results = self.send_alert(alert)
        
        return alert, results
    
    def alert_drift_detected(
        self,
        drift_type: str,
        drift_share: float,
        drifted_features: List[str],
        threshold: float
    ) -> Alert:
        """
        Create and send a drift detection alert.
        
        Args:
            drift_type: Type of drift (data, prediction)
            drift_share: Share of drifted features
            drifted_features: List of drifted feature names
            threshold: Drift threshold used
        
        Returns:
            Created Alert
        """
        severity = "critical" if drift_share > 0.3 else "warning"
        
        alert, _ = self.create_and_send(
            alert_type="drift",
            severity=severity,
            title=f"{drift_type.title()} Drift Detected",
            message=(
                f"{drift_type.title()} drift detected with {drift_share:.1%} of features drifted.\n"
                f"Drifted features: {', '.join(drifted_features[:5])}"
                f"{'...' if len(drifted_features) > 5 else ''}"
            ),
            source="drift_detector",
            metric_name="drift_share",
            current_value=drift_share,
            threshold_value=threshold
        )
        
        return alert
    
    def alert_performance_degradation(
        self,
        metric_name: str,
        current_value: float,
        threshold: float
    ) -> Alert:
        """
        Create and send a performance degradation alert.
        
        Args:
            metric_name: Name of the degraded metric
            current_value: Current metric value
            threshold: Minimum threshold
        
        Returns:
            Created Alert
        """
        severity = "critical" if metric_name == "recall" else "warning"
        
        alert, _ = self.create_and_send(
            alert_type="performance",
            severity=severity,
            title=f"Performance Degradation: {metric_name}",
            message=(
                f"{metric_name} has dropped to {current_value:.4f}, "
                f"which is below the threshold of {threshold:.4f}."
            ),
            source="performance_monitor",
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold
        )
        
        return alert
    
    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> Optional[Alert]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who acknowledged
        
        Returns:
            Updated Alert or None
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = "acknowledged"
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now().isoformat()
                alert.updated_at = datetime.now().isoformat()
                return alert
        
        return None
    
    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of alert to resolve
        
        Returns:
            Updated Alert or None
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = "resolved"
                alert.resolved_at = datetime.now().isoformat()
                alert.updated_at = datetime.now().isoformat()
                return alert
        
        return None
    
    def get_open_alerts(self) -> List[Alert]:
        """Get all open alerts."""
        return [a for a in self.alerts if a.status == "open"]
    
    def get_alerts_by_severity(self, severity: str) -> List[Alert]:
        """Get alerts by severity level."""
        return [a for a in self.alerts if a.severity == severity]
    
    def get_alerts_summary(self) -> Dict:
        """Get summary of alerts."""
        return {
            "total": len(self.alerts),
            "open": len([a for a in self.alerts if a.status == "open"]),
            "acknowledged": len([a for a in self.alerts if a.status == "acknowledged"]),
            "resolved": len([a for a in self.alerts if a.status == "resolved"]),
            "by_severity": {
                "critical": len([a for a in self.alerts if a.severity == "critical"]),
                "warning": len([a for a in self.alerts if a.severity == "warning"]),
                "info": len([a for a in self.alerts if a.severity == "info"])
            },
            "by_type": {
                "drift": len([a for a in self.alerts if a.alert_type == "drift"]),
                "performance": len([a for a in self.alerts if a.alert_type == "performance"]),
                "system": len([a for a in self.alerts if a.alert_type == "system"])
            }
        }


# Need to import Tuple
from typing import Tuple

# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get singleton alert manager instance."""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
    
    return _alert_manager