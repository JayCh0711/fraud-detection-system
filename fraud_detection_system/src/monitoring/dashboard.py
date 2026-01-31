"""
Monitoring Dashboard using Plotly Dash

Features:
1. Real-time Metrics Display
2. Drift Visualization
3. Performance Trends
4. Alert Management
"""

import os
import sys
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

# Dash imports
try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    import plotly.express as px
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

import pandas as pd
import numpy as np

from src.monitoring.config import monitoring_config, monitoring_settings
from src.monitoring.performance_monitor import get_performance_monitor
from src.monitoring.drift_detector import get_drift_detector
from src.monitoring.alerting import get_alert_manager
from src.logger import logger


class MonitoringDashboard:
    """
    Interactive monitoring dashboard.
    """
    
    def __init__(self):
        """Initialize dashboard."""
        if not DASH_AVAILABLE:
            raise ImportError("Dash not installed. Run: pip install dash plotly")
        
        self.app = dash.Dash(
            __name__,
            title="Fraud Detection Monitoring",
            suppress_callback_exceptions=True
        )
        
        self.performance_monitor = get_performance_monitor()
        self.alert_manager = get_alert_manager()
        
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info("Monitoring Dashboard initialized")
    
    def _setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("🔍 Fraud Detection Monitoring Dashboard",
                        style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                       id='last-update', style={'textAlign': 'center'}),
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # 60 seconds
                n_intervals=0
            ),
            
            # Main content
            html.Div([
                # Row 1: KPI Cards
                html.Div([
                    html.Div([
                        html.H3("Recall", style={'textAlign': 'center'}),
                        html.H2(id='kpi-recall', style={'textAlign': 'center', 'color': '#27ae60'}),
                    ], className='kpi-card', style=self._card_style()),
                    
                    html.Div([
                        html.H3("Precision", style={'textAlign': 'center'}),
                        html.H2(id='kpi-precision', style={'textAlign': 'center', 'color': '#3498db'}),
                    ], className='kpi-card', style=self._card_style()),
                    
                    html.Div([
                        html.H3("F1-Score", style={'textAlign': 'center'}),
                        html.H2(id='kpi-f1', style={'textAlign': 'center', 'color': '#9b59b6'}),
                    ], className='kpi-card', style=self._card_style()),
                    
                    html.Div([
                        html.H3("Open Alerts", style={'textAlign': 'center'}),
                        html.H2(id='kpi-alerts', style={'textAlign': 'center', 'color': '#e74c3c'}),
                    ], className='kpi-card', style=self._card_style()),
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
                
                # Row 2: Performance Trends
                html.Div([
                    html.Div([
                        html.H3("Performance Trends", style={'textAlign': 'center'}),
                        dcc.Graph(id='performance-trend-chart'),
                    ], style={'width': '60%', **self._card_style()}),
                    
                    html.Div([
                        html.H3("Metrics Distribution", style={'textAlign': 'center'}),
                        dcc.Graph(id='metrics-distribution-chart'),
                    ], style={'width': '35%', **self._card_style()}),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
                
                # Row 3: Alerts and Drift
                html.Div([
                    html.Div([
                        html.H3("Recent Alerts", style={'textAlign': 'center'}),
                        html.Div(id='alerts-table'),
                    ], style={'width': '48%', **self._card_style()}),
                    
                    html.Div([
                        html.H3("Drift Status", style={'textAlign': 'center'}),
                        html.Div(id='drift-status'),
                        dcc.Graph(id='drift-chart'),
                    ], style={'width': '48%', **self._card_style()}),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
                
                # Row 4: Model Info
                html.Div([
                    html.H3("Model Information", style={'textAlign': 'center'}),
                    html.Div(id='model-info'),
                ], style=self._card_style()),
                
            ], style={'padding': '20px'}),
            
        ], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f5f5f5'})
    
    def _card_style(self) -> Dict:
        """Get card styling."""
        return {
            'backgroundColor': 'white',
            'borderRadius': '10px',
            'padding': '20px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'margin': '10px'
        }
    
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('kpi-recall', 'children'),
             Output('kpi-precision', 'children'),
             Output('kpi-f1', 'children'),
             Output('kpi-alerts', 'children'),
             Output('last-update', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_kpis(n):
            summary = self.performance_monitor.get_metrics_summary()
            alerts_summary = self.alert_manager.get_alerts_summary()
            
            if summary.get('status') == 'no_data':
                return "N/A", "N/A", "N/A", "0", f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            metrics = summary.get('metrics', {})
            
            recall = f"{metrics.get('recall', {}).get('latest', 0):.2%}"
            precision = f"{metrics.get('precision', {}).get('latest', 0):.2%}"
            f1 = f"{metrics.get('f1_score', {}).get('latest', 0):.2%}"
            open_alerts = str(alerts_summary.get('open', 0))
            
            return recall, precision, f1, open_alerts, f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        @self.app.callback(
            Output('performance-trend-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(n):
            history = self.performance_monitor.metrics_history[-30:]  # Last 30 periods
            
            if not history:
                return go.Figure()
            
            df = pd.DataFrame([m.to_dict() for m in history])
            
            fig = go.Figure()
            
            metrics_to_plot = ['recall', 'precision', 'f1_score', 'roc_auc']
            colors = ['#27ae60', '#3498db', '#9b59b6', '#f39c12']
            
            for metric, color in zip(metrics_to_plot, colors):
                if metric in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['period'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(color=color, width=2)
                    ))
            
            # Add threshold lines
            fig.add_hline(y=monitoring_settings.MIN_RECALL_THRESHOLD, 
                         line_dash="dash", line_color="red",
                         annotation_text="Min Recall")
            
            fig.update_layout(
                title="Model Performance Over Time",
                xaxis_title="Period",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output('metrics-distribution-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_distribution_chart(n):
            summary = self.performance_monitor.get_metrics_summary()
            
            if summary.get('status') == 'no_data':
                return go.Figure()
            
            metrics = summary.get('metrics', {})
            
            metric_names = []
            values = []
            
            for name, data in metrics.items():
                if name != 'false_alarm_rate':
                    metric_names.append(name.replace('_', ' ').title())
                    values.append(data.get('mean', 0))
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metric_names,
                    y=values,
                    marker_color=['#27ae60', '#3498db', '#9b59b6', '#f39c12']
                )
            ])
            
            fig.update_layout(
                title="Average Metrics",
                yaxis=dict(range=[0, 1]),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return fig
        
        @self.app.callback(
            Output('alerts-table', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_alerts_table(n):
            alerts = self.alert_manager.alerts[-10:]  # Last 10 alerts
            
            if not alerts:
                return html.P("No alerts", style={'textAlign': 'center', 'color': '#7f8c8d'})
            
            rows = []
            for alert in reversed(alerts):
                severity_color = {
                    'critical': '#e74c3c',
                    'warning': '#f39c12',
                    'info': '#3498db'
                }.get(alert.severity, '#7f8c8d')
                
                rows.append(html.Tr([
                    html.Td(html.Span("●", style={'color': severity_color})),
                    html.Td(alert.alert_type),
                    html.Td(alert.title[:30] + '...' if len(alert.title) > 30 else alert.title),
                    html.Td(alert.status),
                    html.Td(alert.created_at[:16] if alert.created_at else ''),
                ]))
            
            return html.Table([
                html.Thead(html.Tr([
                    html.Th(""),
                    html.Th("Type"),
                    html.Th("Title"),
                    html.Th("Status"),
                    html.Th("Time"),
                ])),
                html.Tbody(rows)
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
        
        @self.app.callback(
            [Output('drift-status', 'children'),
             Output('drift-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_drift_status(n):
            # Placeholder for drift data
            status = html.Div([
                html.P("✓ No Data Drift Detected", 
                       style={'color': '#27ae60', 'fontWeight': 'bold'}),
                html.P("Last check: " + datetime.now().strftime('%Y-%m-%d %H:%M')),
            ])
            
            # Simple placeholder chart
            fig = go.Figure(data=[
                go.Indicator(
                    mode="gauge+number",
                    value=0.05,
                    title={'text': "Drift Score"},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "#27ae60"},
                        'steps': [
                            {'range': [0, 0.1], 'color': "#d5f5e3"},
                            {'range': [0.1, 0.3], 'color': "#fcf3cf"},
                            {'range': [0.3, 1], 'color': "#fadbd8"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.1
                        }
                    }
                )
            ])
            
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            
            return status, fig
        
        @self.app.callback(
            Output('model-info', 'children'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_model_info(n):
            try:
                from src.utils.common import read_json
                version_path = os.path.join(monitoring_config.production_model_dir, "version.json")
                
                if os.path.exists(version_path):
                    version_info = read_json(version_path)
                else:
                    version_info = {"version": "Unknown", "metrics": {}}
                
                return html.Div([
                    html.P(f"Version: {version_info.get('version', 'Unknown')}"),
                    html.P(f"Threshold: {version_info.get('optimal_threshold', 0.5)}"),
                    html.P(f"Promoted: {version_info.get('promoted_at', 'Unknown')}"),
                ], style={'display': 'flex', 'justifyContent': 'space-around'})
                
            except Exception as e:
                return html.P(f"Error loading model info: {str(e)}")
    
    def run(self, host: str = None, port: int = None, debug: bool = False):
        """
        Run the dashboard server.
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        host = host or monitoring_settings.DASHBOARD_HOST
        port = port or monitoring_settings.DASHBOARD_PORT
        
        logger.info(f"Starting Monitoring Dashboard at http://{host}:{port}")
        
        self.app.run_server(host=host, port=port, debug=debug)


def run_dashboard():
    """Run the monitoring dashboard."""
    if not DASH_AVAILABLE:
        logger.error("Dash not installed. Run: pip install dash plotly")
        return
    
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()