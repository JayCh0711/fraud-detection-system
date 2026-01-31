"""
Monitoring Application Entry Point

Run monitoring services:
1. Drift Detection
2. Performance Monitoring
3. Dashboard
4. MLflow Server
"""

import argparse
import sys
import os
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.monitoring.config import monitoring_config, monitoring_settings
from src.monitoring.drift_detector import get_drift_detector
from src.monitoring.performance_monitor import get_performance_monitor
from src.monitoring.alerting import get_alert_manager
from src.monitoring.mlflow_tracker import get_mlflow_tracker
from src.logger import logger


def run_drift_check(data_path: str = None):
    """Run drift detection check."""
    logger.info("Running drift detection...")
    
    import pandas as pd
    
    detector = get_drift_detector()
    detector.load_reference_data()
    
    if data_path and os.path.exists(data_path):
        current_data = pd.read_csv(data_path)
    else:
        # Use test data as current data for demo
        test_path = os.path.join(
            "artifacts", "data_ingestion", "processed", "test.csv"
        )
        if os.path.exists(test_path):
            current_data = pd.read_csv(test_path)
        else:
            logger.error("No data available for drift detection")
            return
    
    result = detector.detect_data_drift(current_data)
    
    if result.is_drift_detected:
        alert_manager = get_alert_manager()
        alert_manager.alert_drift_detected(
            drift_type="data",
            drift_share=result.drift_share,
            drifted_features=result.drifted_features,
            threshold=result.threshold_used
        )
    
    logger.info(f"Drift check complete. Drift detected: {result.is_drift_detected}")


def run_performance_check():
    """Run performance monitoring check."""
    logger.info("Running performance check...")
    
    import numpy as np
    
    monitor = get_performance_monitor()
    
    # Load test data and predictions for demo
    try:
        y_test = np.load("artifacts/data_transformation/test_target.npy")
        
        # Load model and make predictions
        from src.pipeline.prediction_pipeline import PredictionPipeline
        pipeline = PredictionPipeline()
        pipeline.initialize()
        
        X_test = np.load("artifacts/data_transformation/test_transformed.npy")
        y_pred_proba = pipeline.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= pipeline.optimal_threshold).astype(int)
        
        # Calculate and log metrics
        metrics = monitor.calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            threshold=pipeline.optimal_threshold,
            period=datetime.now().strftime("%Y-%m-%d")
        )
        
        alerts = monitor.log_metrics(metrics)
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} performance alerts")
        
    except Exception as e:
        logger.error(f"Performance check failed: {str(e)}")


def run_monitoring_loop(interval_minutes: int = 60):
    """Run continuous monitoring loop."""
    logger.info(f"Starting monitoring loop (interval: {interval_minutes} min)")
    
    while True:
        try:
            run_drift_check()
            run_performance_check()
            
            logger.info(f"Next check in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("Monitoring loop stopped")
            break
        except Exception as e:
            logger.error(f"Monitoring error: {str(e)}")
            time.sleep(60)  # Wait 1 minute before retry


def run_dashboard():
    """Run monitoring dashboard."""
    try:
        from src.monitoring.dashboard import MonitoringDashboard
        dashboard = MonitoringDashboard()
        dashboard.run()
    except ImportError as e:
        logger.error(f"Dashboard dependencies not installed: {e}")
        logger.error("Run: pip install dash plotly")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fraud Detection Monitoring")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['drift', 'performance', 'loop', 'dashboard', 'all'],
        default='all',
        help='Monitoring mode'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Monitoring interval in minutes (for loop mode)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to current data for drift detection'
    )
    
    args = parser.parse_args()
    
    logger.info(f"{'='*60}")
    logger.info("FRAUD DETECTION MONITORING")
    logger.info(f"{'='*60}")
    logger.info(f"Mode: {args.mode}")
    
    if args.mode == 'drift':
        run_drift_check(args.data_path)
    
    elif args.mode == 'performance':
        run_performance_check()
    
    elif args.mode == 'loop':
        run_monitoring_loop(args.interval)
    
    elif args.mode == 'dashboard':
        run_dashboard()
    
    elif args.mode == 'all':
        # Run dashboard in separate thread
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Run monitoring loop in main thread
        run_monitoring_loop(args.interval)


if __name__ == "__main__":
    main()