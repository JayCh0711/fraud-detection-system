"""
Model Performance Monitoring

Tracks:
1. Classification Metrics over time
2. Business Metrics
3. Threshold Analysis
4. Performance Degradation Detection
"""

import os
import sys
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

from src.monitoring.config import monitoring_config, monitoring_settings
from src.logger import logger
from src.utils.common import write_json, read_json, create_directories


@dataclass
class PerformanceMetrics:
    """Model performance metrics at a point in time"""
    
    # Classification Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    
    # Confusion Matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Business Metrics
    fraud_detection_rate: float = 0.0
    false_alarm_rate: float = 0.0
    
    # Sample Info
    total_samples: int = 0
    fraud_samples: int = 0
    legitimate_samples: int = 0
    
    # Threshold
    threshold_used: float = 0.5
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    period: str = ""  # e.g., "2024-01-15", "2024-W02", "2024-01"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def meets_thresholds(self, config: monitoring_settings = None) -> Dict[str, bool]:
        """Check if metrics meet minimum thresholds."""
        config = config or monitoring_settings
        
        return {
            'recall': self.recall >= config.MIN_RECALL_THRESHOLD,
            'precision': self.precision >= config.MIN_PRECISION_THRESHOLD,
            'f1_score': self.f1_score >= config.MIN_F1_THRESHOLD,
            'false_alarm_rate': self.false_alarm_rate <= config.MAX_FALSE_ALARM_RATE
        }


@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""
    
    alert_id: str
    alert_type: str  # "threshold_breach", "degradation", "anomaly"
    severity: str  # "warning", "critical"
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceMonitor:
    """
    Model Performance Monitoring System.
    
    Tracks model performance over time and detects degradation.
    """
    
    def __init__(self, config: monitoring_config = None):
        """
        Initialize Performance Monitor.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or monitoring_config
        self.settings = monitoring_settings
        
        self.metrics_history: List[PerformanceMetrics] = []
        self.alerts: List[PerformanceAlert] = []
        
        # Create directories
        create_directories([
            self.config.performance_reports_dir,
            self.config.alerts_dir
        ])
        
        # Load historical metrics
        self._load_history()
        
        logger.info("Performance Monitor initialized")
    
    def _load_history(self) -> None:
        """Load historical metrics from file."""
        history_path = os.path.join(
            self.config.performance_reports_dir,
            "metrics_history.json"
        )
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                
                self.metrics_history = [
                    PerformanceMetrics(**m) for m in history_data
                ]
                
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
                
            except Exception as e:
                logger.warning(f"Could not load history: {str(e)}")
    
    def _save_history(self) -> None:
        """Save metrics history to file."""
        history_path = os.path.join(
            self.config.performance_reports_dir,
            "metrics_history.json"
        )
        
        history_data = [m.to_dict() for m in self.metrics_history]
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        period: str = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            threshold: Classification threshold
            period: Time period identifier
        
        Returns:
            PerformanceMetrics object
        """
        # Calculate metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = PerformanceMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, y_pred_proba)) if len(np.unique(y_true)) > 1 else 0.0,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            fraud_detection_rate=float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            false_alarm_rate=float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            total_samples=len(y_true),
            fraud_samples=int(y_true.sum()),
            legitimate_samples=int(len(y_true) - y_true.sum()),
            threshold_used=threshold,
            period=period or datetime.now().strftime("%Y-%m-%d")
        )
        
        return metrics
    
    def log_metrics(
        self,
        metrics: PerformanceMetrics,
        check_thresholds: bool = True
    ) -> List[PerformanceAlert]:
        """
        Log metrics and check for threshold breaches.
        
        Args:
            metrics: Performance metrics to log
            check_thresholds: Whether to check for threshold breaches
        
        Returns:
            List of alerts generated
        """
        logger.info("Logging performance metrics...")
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Save history
        self._save_history()
        
        # Save individual report
        report_path = os.path.join(
            self.config.performance_reports_dir,
            f"metrics_{metrics.period.replace('-', '')}_{datetime.now().strftime('%H%M%S')}.json"
        )
        write_json(report_path, metrics.to_dict())
        
        # Log summary
        logger.info(f"Performance Metrics ({metrics.period}):")
        logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1-Score:  {metrics.f1_score:.4f}")
        logger.info(f"  ROC-AUC:   {metrics.roc_auc:.4f}")
        logger.info(f"  Samples:   {metrics.total_samples}")
        
        # Check thresholds
        alerts = []
        if check_thresholds:
            alerts = self._check_thresholds(metrics)
        
        return alerts
    
    def _check_thresholds(
        self,
        metrics: PerformanceMetrics
    ) -> List[PerformanceAlert]:
        """
        Check if metrics breach thresholds.
        
        Args:
            metrics: Metrics to check
        
        Returns:
            List of alerts
        """
        alerts = []
        import uuid
        
        threshold_checks = [
            ('recall', metrics.recall, self.settings.MIN_RECALL_THRESHOLD, '>='),
            ('precision', metrics.precision, self.settings.MIN_PRECISION_THRESHOLD, '>='),
            ('f1_score', metrics.f1_score, self.settings.MIN_F1_THRESHOLD, '>='),
            ('false_alarm_rate', metrics.false_alarm_rate, self.settings.MAX_FALSE_ALARM_RATE, '<='),
        ]
        
        for metric_name, value, threshold, direction in threshold_checks:
            breach = (value < threshold) if direction == '>=' else (value > threshold)
            
            if breach:
                severity = "critical" if metric_name == "recall" else "warning"
                
                alert = PerformanceAlert(
                    alert_id=f"PERF_{uuid.uuid4().hex[:8].upper()}",
                    alert_type="threshold_breach",
                    severity=severity,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=threshold,
                    message=f"{metric_name} ({value:.4f}) breached threshold ({direction} {threshold})"
                )
                
                alerts.append(alert)
                self.alerts.append(alert)
                
                logger.warning(f"⚠️ PERFORMANCE ALERT: {alert.message}")
        
        # Save alerts
        if alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def _save_alerts(self, alerts: List[PerformanceAlert]) -> None:
        """Save alerts to file."""
        for alert in alerts:
            alert_path = os.path.join(
                self.config.alerts_dir,
                f"{alert.alert_id}.json"
            )
            write_json(alert_path, alert.to_dict())
    
    def detect_degradation(
        self,
        window_size: int = 7,
        degradation_threshold: float = 0.05
    ) -> Dict:
        """
        Detect performance degradation over time.
        
        Args:
            window_size: Number of recent periods to compare
            degradation_threshold: Minimum change to consider degradation
        
        Returns:
            Degradation analysis dictionary
        """
        logger.info("Checking for performance degradation...")
        
        if len(self.metrics_history) < window_size:
            return {
                "status": "insufficient_data",
                "message": f"Need at least {window_size} data points"
            }
        
        # Get recent and historical metrics
        recent = self.metrics_history[-window_size:]
        historical = self.metrics_history[:-window_size] if len(self.metrics_history) > window_size else recent
        
        # Calculate average metrics
        metrics_to_check = ['recall', 'precision', 'f1_score', 'roc_auc']
        
        degradation_detected = False
        degradation_details = {}
        
        for metric in metrics_to_check:
            recent_avg = np.mean([getattr(m, metric) for m in recent])
            historical_avg = np.mean([getattr(m, metric) for m in historical])
            
            change = recent_avg - historical_avg
            pct_change = (change / historical_avg * 100) if historical_avg > 0 else 0
            
            is_degraded = change < -degradation_threshold
            
            degradation_details[metric] = {
                'recent_avg': round(recent_avg, 4),
                'historical_avg': round(historical_avg, 4),
                'change': round(change, 4),
                'pct_change': round(pct_change, 2),
                'is_degraded': is_degraded
            }
            
            if is_degraded:
                degradation_detected = True
                logger.warning(f"⚠️ {metric} degradation: {pct_change:.2f}%")
        
        result = {
            "status": "degradation_detected" if degradation_detected else "stable",
            "window_size": window_size,
            "threshold": degradation_threshold,
            "details": degradation_details,
            "timestamp": datetime.now().isoformat()
        }
        
        if degradation_detected:
            logger.warning("Performance degradation detected!")
        else:
            logger.info("✓ No significant performance degradation")
        
        return result
    
    def get_metrics_summary(
        self,
        last_n_periods: int = 30
    ) -> Dict:
        """
        Get summary of recent metrics.
        
        Args:
            last_n_periods: Number of recent periods to summarize
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent = self.metrics_history[-last_n_periods:]
        
        metrics_to_summarize = ['recall', 'precision', 'f1_score', 'roc_auc', 'false_alarm_rate']
        
        summary = {
            "period_count": len(recent),
            "date_range": {
                "start": recent[0].period if recent else None,
                "end": recent[-1].period if recent else None
            },
            "total_samples": sum(m.total_samples for m in recent),
            "total_frauds": sum(m.fraud_samples for m in recent),
            "metrics": {}
        }
        
        for metric in metrics_to_summarize:
            values = [getattr(m, metric) for m in recent]
            summary["metrics"][metric] = {
                "mean": round(np.mean(values), 4),
                "std": round(np.std(values), 4),
                "min": round(np.min(values), 4),
                "max": round(np.max(values), 4),
                "latest": round(values[-1], 4) if values else 0
            }
        
        return summary
    
    def generate_performance_report(
        self,
        output_path: str = None
    ) -> str:
        """
        Generate performance report.
        
        Args:
            output_path: Path for report file
        
        Returns:
            Path to generated report
        """
        logger.info("Generating performance report...")
        
        summary = self.get_metrics_summary()
        degradation = self.detect_degradation()
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": summary,
            "degradation_analysis": degradation,
            "recent_alerts": [a.to_dict() for a in self.alerts[-10:]],
            "thresholds": {
                "min_recall": self.settings.MIN_RECALL_THRESHOLD,
                "min_precision": self.settings.MIN_PRECISION_THRESHOLD,
                "min_f1": self.settings.MIN_F1_THRESHOLD,
                "max_false_alarm_rate": self.settings.MAX_FALSE_ALARM_RATE
            }
        }
        
        if output_path is None:
            output_path = os.path.join(
                self.config.performance_reports_dir,
                f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        
        write_json(output_path, report)
        logger.info(f"Performance report saved: {output_path}")
        
        return output_path


# Singleton instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor