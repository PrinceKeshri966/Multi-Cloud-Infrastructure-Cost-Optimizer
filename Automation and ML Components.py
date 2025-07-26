# ml_optimizer.py - Machine Learning components for cost optimization
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CostOptimizationML:
    def __init__(self):
        self.anomaly_detector = None
        self.cost_predictor = None
        self.resource_clusterer = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        features = df.copy()
        
        # Time-based features
        features['hour'] = pd.to_datetime(features['usage_start_time']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['usage_start_time']).dt.dayofweek
        features['month'] = pd.to_datetime(features['usage_start_time']).dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Cost features
        features['cost_per_hour'] = features['cost'] / features.get('usage_hours', 1)
        features['cost_trend'] = features.groupby(['project_id', 'service_name'])['cost'].pct_change()
        
        # Resource utilization features
        if 'cpu_utilization' in features.columns:
            features['cpu_utilization_category'] = pd.cut(
                features['cpu_utilization'], 
                bins=[0, 20, 50, 80, 100], 
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        # Lag features for time series
        for lag in [1, 7, 30]:
            features[f'cost_lag_{lag}'] = features.groupby(['project_id', 'service_name'])['cost'].shift(lag)
            features[f'usage_lag_{lag}'] = features.groupby(['project_id', 'service_name'])['usage_amount'].shift(lag)
        
        # Statistical features
        features['cost_rolling_mean_7d'] = features.groupby(['project_id', 'service_name'])['cost'].rolling(7).mean().reset_index(0, drop=True)
        features['cost_rolling_std_7d'] = features.groupby(['project_id', 'service_name'])['cost'].rolling(7).std().reset_index(0, drop=True)
        
        return features
    
    def train_anomaly_detector(self, df: pd.DataFrame) -> Dict:
        """Train anomaly detection model"""
        logger.info("Training anomaly detection model...")
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Select numerical features for anomaly detection
        numerical_features = [
            'cost', 'usage_amount', 'hour', 'day_of_week', 'month',
            'cost_per_hour', 'cost_trend', 'cost_rolling_mean_7d', 'cost_rolling_std_7d'
        ]
        
        # Filter available features
        available_features = [f for f in numerical_features if f in features_df.columns]
        
        # Handle missing values
        X = features_df[available_features].fillna(features_df[available_features].median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.anomaly_detector = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_jobs=-1
        )
        
        anomalies = self.anomaly_detector.fit_predict(X_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        
        # Calculate performance metrics
        anomaly_ratio = (anomalies == -1).sum() / len(anomalies)
        
        results = {
            'model_type': 'IsolationForest',
            'features_used': available_features,
            'anomaly_ratio': anomaly_ratio,
            'score_range': {
                'min': float(anomaly_scores.min()),
                'max': float(anomaly_scores.max()),
                'mean': float(anomaly_scores.mean())
            },
            'training_samples': len(X),
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Anomaly detection model trained. Anomaly ratio: {anomaly_ratio:.3f}")
        return results
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in cost data"""
        if self.anomaly_detector is None:
            raise ValueError("Anomaly detector not trained. Call train_anomaly_detector first.")
        
        features_df = self.prepare_features(df)
        
        numerical_features = [
            'cost', 'usage_amount', 'hour', 'day_of_week', 'month',
            'cost_per_hour', 'cost_trend', 'cost_rolling_mean_7d', 'cost_rolling_std_7d'
        ]
        
        available_features = [f for f in numerical_features if f in features_df.columns]
        X = features_df[available_features].fillna(features_df[available_features].median())
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies
        anomalies = self.anomaly_detector.predict(X_scaled)
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['is_anomaly'] = anomalies == -1
        result_df['anomaly_score'] = anomaly_scores
        result_df['anomaly_severity'] = pd.cut(
            anomaly_scores, 
            bins=[-float('inf'), -0.2, -0.1, 0, float('inf')],
            labels=['Critical', 'High', 'Medium', 'Normal']
        )
        
        return result_df[result_df['is_anomaly']]
    
    def train_cost_predictor(self, df: pd.DataFrame) -> Dict:
        """Train cost prediction model"""
        logger.info("Training cost prediction model...")
        
        features_df = self.prepare_features(df)
        
        # Features for prediction
        feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'usage_amount', 'cost_lag_1', 'cost_lag_7', 'cost_lag_30',
            'cost_rolling_mean_7d', 'cost_rolling_std_7d'
        ]
        
        # Filter available features and handle missing values
        available_features = [f for f in feature_columns if f in features_df.columns]
        
        # Prepare data
        X = features_df[available_features].fillna(method='ffill').fillna(0)
        y = features_df['cost']
        
        # Remove rows with NaN target
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training cost predictor")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Time series split
        )
        
        # Train Random Forest
        self.cost_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.cost_predictor.fit(X_train, y_train)
        self.feature_columns = available_features
        
        # Evaluate model
        y_pred = self.cost_predictor.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(
            available_features,
            self.cost_predictor.feature_importances_
        ))
        
        results = {
            'model_type': 'RandomForestRegressor',
            'features_used': available_features,
            'performance': {
                'mae': float(mae),
                'r2_score': float(r2),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info(f"Cost predictor trained. MAE: {mae:.2f}, R²: {r2:.3f}")
        return results
    
    def predict_costs(self, df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """Predict future costs"""
        if self.cost_predictor is None:
            raise ValueError("Cost predictor not trained. Call train_cost_predictor first.")
        
        predictions = []
        
        # Group by project and service for individual predictions
        for (project_id, service_name), group in df.groupby(['project_id', 'service_name']):
            group_features = self.prepare_features(group)
            
            # Get latest data point
            latest_data = group_features.iloc[-1:].copy()
            
            for day in range(1, days_ahead + 1):
                # Update time features
                future_date = pd.to_datetime(latest_data['usage_start_time'].iloc[0]) + timedelta(days=day)
                latest_data['hour'] = future_date.hour
                latest_data['day_of_week'] = future_date.dayofweek
                latest_data['month'] = future_date.month
                latest_data['is_weekend'] = int(future_date.dayofweek in [5, 6])
                
                # Prepare features for prediction
                X_pred = latest_data[self.feature_columns].fillna(method='ffill').fillna(0)
                
                # Make prediction
                predicted_cost = self.cost_predictor.predict(X_pred)[0]
                
                predictions.append({
                    'project_id': project_id,
                    'service_name': service_name,
                    'prediction_date': future_date.date(),
                    'predicted_cost': predicted_cost,
                    'confidence_interval_lower': predicted_cost * 0.8,  # Simplified CI
                    'confidence_interval_upper': predicted_cost * 1.2
                })
        
        return pd.DataFrame(predictions)
    
    def cluster_resources(self, df: pd.DataFrame) -> Dict:
        """Cluster resources based on usage patterns"""
        logger.info("Clustering resources by usage patterns...")
        
        # Aggregate resource usage patterns
        resource_features = df.groupby(['project_id', 'service_name']).agg({
            'cost': ['mean', 'std', 'max', 'min'],
            'usage_amount': ['mean', 'std', 'max'],
            'cpu_utilization': 'mean' if 'cpu_utilization' in df.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        resource_features.columns = ['_'.join(col).strip() for col in resource_features.columns]
        resource_features = resource_features.fillna(0)
        
        # Scale features
        X_scaled = StandardScaler().fit_transform(resource_features)
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        K_range = range(2, min(11, len(resource_features)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Choose optimal k (simplified elbow detection)
        optimal_k = 5 if len(K_range) >= 4 else max(K_range)
        
        # Final clustering
        self.resource_clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.resource_clusterer.fit_predict(X_scaled)
        
        # Add cluster labels to resources
        resource_features['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_data = resource_features[resource_features['cluster'] == cluster_id]
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'characteristics': {
                    'avg_cost': float(cluster_data['cost_mean'].mean()),
                    'cost_volatility': float(cluster_data['cost_std'].mean()),
                    'avg_usage': float(cluster_data['usage_amount_mean'].mean()),
                    'resources': cluster_data.index.tolist()
                }
            }
        
        results = {
            'optimal_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'total_resources': len(resource_features),
            'clustered_at': datetime.now().isoformat()
        }
        
        logger.info(f"Resources clustered into {optimal_k} groups")
        return results
    
    def generate_optimization_insights(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive optimization insights using ML"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'data_period': {
                'start': df['usage_start_time'].min(),
                'end': df['usage_start_time'].max(),
                'total_records': len(df)
            }
        }
        
        try:
            # Train models
            anomaly_results = self.train_anomaly_detector(df)
            cost_prediction_results = self.train_cost_predictor(df)
            clustering_results = self.cluster_resources(df)
            
            # Detect current anomalies
            anomalies_df = self.detect_anomalies(df)
            
            # Generate cost predictions
            cost_predictions = self.predict_costs(df, days_ahead=30)
            
            insights.update({
                'anomaly_detection': {
                    'model_info': anomaly_results,
                    'current_anomalies': len(anomalies_df),
                    'anomaly_details': anomalies_df.to_dict('records') if len(anomalies_df) < 50 else []
                },
                'cost_prediction': {
                    'model_info': cost_prediction_results,
                    'predictions': cost_predictions.to_dict('records')
                },
                'resource_clustering': clustering_results,
                'optimization_recommendations': self._generate_ml_recommendations(df, anomalies_df, clustering_results)
            })
            
        except Exception as e:
            logger.error(f"Error generating ML insights: {str(e)}")
            insights['error'] = str(e)
        
        return insights
    
    def _generate_ml_recommendations(self, df: pd.DataFrame, anomalies_df: pd.DataFrame, 
                                   clustering_results: Dict) -> List[Dict]:
        """Generate ML-based optimization recommendations"""
        recommendations = []
        
        # Anomaly-based recommendations
        if len(anomalies_df) > 0:
            high_cost_anomalies = anomalies_df[anomalies_df['cost'] > anomalies_df['cost'].quantile(0.8)]
            
            for _, anomaly in high_cost_anomalies.iterrows():
                recommendations.append({
                    'type': 'anomaly_investigation',
                    'priority': 'high',
                    'resource': {
                        'project_id': anomaly['project_id'],
                        'service_name': anomaly['service_name']
                    },
                    'description': f"Investigate cost anomaly detected on {anomaly['usage_start_time']}",
                    'potential_impact': f"${anomaly['cost']:.2f}",
                    'confidence': float(abs(anomaly['anomaly_score']))
                })
        
        # Cluster-based recommendations
        for cluster_id, cluster_info in clustering_results.get('cluster_analysis', {}).items():
            if cluster_info['characteristics']['cost_volatility'] > 100:  # High volatility
                recommendations.append({
                    'type': 'cost_optimization',
                    'priority': 'medium',
                    'resource': {
                        'cluster': cluster_id,
                        'affected_resources': len(cluster_info['characteristics']['resources'])
                    },
                    'description': f"High cost volatility detected in {cluster_id}",
                    'potential_impact': f"${cluster_info['characteristics']['avg_cost'] * 0.2:.2f} monthly savings",
                    'confidence': 0.7
                })
        
        # Usage pattern recommendations
        service_usage = df.groupby('service_name').agg({
            'cost': 'sum',
            'usage_amount': 'mean'
        }).round(2)
        
        for service, data in service_usage.iterrows():
            if data['cost'] > service_usage['cost'].quantile(0.9):  # Top 10% expensive services
                recommendations.append({
                    'type': 'service_optimization',
                    'priority': 'high',
                    'resource': {
                        'service_name': service
                    },
                    'description': f"Review {service} - high cost service",
                    'potential_impact': f"${data['cost'] * 0.15:.2f} potential monthly savings",
                    'confidence': 0.8
                })
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    def save_models(self, filepath_prefix: str):
        """Save trained models to disk"""
        if self.anomaly_detector:
            joblib.dump(self.anomaly_detector, f"{filepath_prefix}_anomaly_detector.pkl")
            joblib.dump(self.scaler, f"{filepath_prefix}_scaler.pkl")
        
        if self.cost_predictor:
            joblib.dump(self.cost_predictor, f"{filepath_prefix}_cost_predictor.pkl")
            
            # Save feature columns
            with open(f"{filepath_prefix}_features.json", 'w') as f:
                json.dump(self.feature_columns, f)
        
        if self.resource_clusterer:
            joblib.dump(self.resource_clusterer, f"{filepath_prefix}_clusterer.pkl")
        
        logger.info(f"Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str):
        """Load trained models from disk"""
        try:
            self.anomaly_detector = joblib.load(f"{filepath_prefix}_anomaly_detector.pkl")
            self.scaler = joblib.load(f"{filepath_prefix}_scaler.pkl")
            logger.info("Anomaly detection model loaded")
        except FileNotFoundError:
            logger.warning("Anomaly detection model not found")
        
        try:
            self.cost_predictor = joblib.load(f"{filepath_prefix}_cost_predictor.pkl")
            with open(f"{filepath_prefix}_features.json", 'r') as f:
                self.feature_columns = json.load(f)
            logger.info("Cost prediction model loaded")
        except FileNotFoundError:
            logger.warning("Cost prediction model not found")
        
        try:
            self.resource_clusterer = joblib.load(f"{filepath_prefix}_clusterer.pkl")
            logger.info("Resource clustering model loaded")
        except FileNotFoundError:
            logger.warning("Resource clustering model not found")

# automation_engine.py - Terraform automation for cost optimization
import subprocess
import json
import os
import tempfile
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TerraformAutomation:
    def __init__(self, project_id: str, credentials_path: Optional[str] = None):
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.terraform_dir = "terraform_automation"
        
        # Ensure terraform directory exists
        os.makedirs(self.terraform_dir, exist_ok=True)
    
    def generate_optimization_terraform(self, recommendations: List[Dict]) -> str:
        """Generate Terraform configuration for optimization recommendations"""
        
        terraform_config = f'''
terraform {{
  required_providers {{
    google = {{
      source  = "hashicorp/google"
      version = "~> 4.0"
    }}
  }}
}}

provider "google" {{
  project = "{self.project_id}"
  region  = "us-central1"
}}

# Committed Use Discounts
'''
        
        # Generate CUD resources
        cud_recommendations = [r for r in recommendations if 'CUD' in r.get('recommended_type', '')]
        
        for i, rec in enumerate(cud_recommendations):
            terraform_config += f'''
resource "google_compute_commitment" "cud_{i}" {{
  name        = "cost-optimization-cud-{i}"
  description = "Automated CUD for {rec['resource_name']}"
  
  plan = "TWELVE_MONTH"
  type = "GENERAL_PURPOSE"
  
  resources {{
    type = "VCPU"
    amount = "10"  # Adjust based on actual usage
  }}
  
  resources {{
    type = "MEMORY"
    amount = "40"  # Adjust based on actual usage
  }}
}}
'''
        
        # Generate preemptible instance configurations
        preemptible_recs = [r for r in recommendations if 'Preemptible' in r.get('recommended_type', '')]
        
        for i, rec in enumerate(preemptible_recs):
            terraform_config += f'''
# Note: This is a template - actual implementation would require instance replacement
resource "google_compute_instance" "preemptible_{i}" {{
  name         = "{rec['resource_name']}-preemptible"
  machine_type = "{rec['recommended_type'].replace(' (Preemptible)', '')}"
  zone         = "us-central1-a"
  
  scheduling {{
    preemptible = true
    automatic_restart = false
    on_host_maintenance = "TERMINATE"
  }}
  
  boot_disk {{
    initialize_params {{
      image = "debian-cloud/debian-11"
    }}
  }}
  
  network_interface {{
    network = "default"
    access_config {{
      // Ephemeral public IP
    }}
  }}
  
  labels = {{
    cost-optimization = "automated"
    original-instance = "{rec['resource_name']}"
  }}
}}
'''
        
        # Add budget alerts
        terraform_config += f'''
resource "google_billing_budget" "cost_optimization_budget" {{
  billing_account = var.billing_account_id
  display_name    = "Cost Optimization Budget"
  
  budget_filter {{
    projects = ["projects/{self.project_id}"]
  }}
  
  amount {{
    specified_amount {{
      currency_code = "USD"
      units         = "1000"  # Adjust based on current spend
    }}
  }}
  
  threshold_rules {{
    threshold_percent = 0.8
    spend_basis      = "CURRENT_SPEND"
  }}
  
  threshold_rules {{
    threshold_percent = 1.0
    spend_basis      = "CURRENT_SPEND"
  }}
  
  all_updates_rule {{
    monitoring_notification_channels = [
      google_monitoring_notification_channel.email.id,
    ]
    disable_default_iam_recipients = false
  }}
}}

resource "google_monitoring_notification_channel" "email" {{
  display_name = "Cost Optimization Alerts"
  type         = "email"
  
  labels = {{
    email_address = var.notification_email
  }}
}}

variable "billing_account_id" {{
  description = "Billing account ID for budget alerts"
  type        = string
}}

variable "notification_email" {{
  description = "Email for cost optimization notifications"
  type        = string
}}
'''
        
        return terraform_config
    
    def apply_optimizations(self, recommendations: List[Dict], 
                          dry_run: bool = True) -> Dict:
        """Apply cost optimizations using Terraform"""
        
        # Generate Terraform configuration
        tf_config = self.generate_optimization_terraform(recommendations)
        
        # Write to file
        tf_file = os.path.join(self.terraform_dir, "main.tf")
        with open(tf_file, 'w') as f:
            f.write(tf_config)
        
        # Create variables file
        tfvars_content = f'''
billing_account_id = "YOUR_BILLING_ACCOUNT_ID"
notification_email = "admin@yourdomain.com"
'''
        
        tfvars_file = os.path.join(self.terraform_dir, "terraform.tfvars")
        with open(tfvars_file, 'w') as f:
            f.write(tfvars_content)
        
        results = {
            'terraform_config_generated': True,
            'config_path': tf_file,
            'dry_run': dry_run,
            'recommendations_processed': len(recommendations)
        }
        
        if not dry_run:
            try:
                # Initialize Terraform
                init_result = subprocess.run(
                    ['terraform', 'init'],
                    cwd=self.terraform_dir,
                    capture_output=True,
                    text=True
                )
                
                if init_result.returncode != 0:
                    results['error'] = f"Terraform init failed: {init_result.stderr}"
                    return results
                
                # Plan
                plan_result = subprocess.run(
                    ['terraform', 'plan', '-out=tfplan'],
                    cwd=self.terraform_dir,
                    capture_output=True,
                    text=True
                )
                
                if plan_result.returncode != 0:
                    results['error'] = f"Terraform plan failed: {plan_result.stderr}"
                    return results
                
                results['plan_output'] = plan_result.stdout
                
                # Apply (with auto-approve for automation)
                apply_result = subprocess.run(
                    ['terraform', 'apply', '-auto-approve', 'tfplan'],
                    cwd=self.terraform_dir,
                    capture_output=True,
                    text=True
                )
                
                if apply_result.returncode != 0:
                    results['error'] = f"Terraform apply failed: {apply_result.stderr}"
                    return results
                
                results['apply_output'] = apply_result.stdout
                results['applied'] = True
                
            except Exception as e:
                results['error'] = f"Terraform execution error: {str(e)}"
        
        return results
    
    def generate_cost_policies(self) -> str:
        """Generate organizational policies for cost control"""
        
        policies_config = f'''
# Organization Policies for Cost Control

resource "google_org_policy_policy" "compute_vm_external_ip_access" {{
  name   = "projects/{self.project_id}/policies/compute.vmExternalIpAccess"
  parent = "projects/{self.project_id}"
  
  spec {{
    rules {{
      deny_all = true
    }}
    
    rules {{
      condition {{
        expression = "resource.matchTag('cost-optimization/external-ip', 'allowed')"
      }}
      allow_all = true
    }}
  }}
}}

resource "google_org_policy_policy" "compute_instance_types" {{
  name   = "projects/{self.project_id}/policies/compute.restrictMachineTypes"
  parent = "projects/{self.project_id}"
  
  spec {{
    rules {{
      values {{
        allowed_values = [
          "e2-micro",
          "e2-small", 
          "e2-medium",
          "e2-standard-2",
          "e2-standard-4",
          "n2-standard-2",
          "n2-standard-4"
        ]
      }}
    }}
  }}
}}

resource "google_org_policy_policy" "storage_location_restriction" {{
  name   = "projects/{self.project_id}/policies/storage.locationRestriction"
  parent = "projects/{self.project_id}"
  
  spec {{
    rules {{
      values {{
        allowed_values = [
          "us-central1",
          "us-east1",
          "us-west1"
        ]
      }}
    }}
  }}
}}
'''
        
        return policies_config

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    sample_data = pd.DataFrame({
        'usage_start_time': pd.date_range('2024-01-01', periods=100, freq='D'),
        'project_id': ['project-1'] * 50 + ['project-2'] * 50,
        'service_name': ['Compute Engine'] * 50 + ['Cloud Storage'] * 50,
        'cost': np.random.uniform(10, 1000, 100),
        'usage_amount': np.random.uniform(1, 100, 100),
        'cpu_utilization': np.random.uniform(0, 100, 100)
    })
    
    # Initialize ML optimizer
    ml_optimizer = CostOptimizationML()
    
    # Generate insights
    insights = ml_optimizer.generate_optimization_insights(sample_data)
    
    print("ML Optimization Insights:")
    print(json.dumps(insights, indent=2, default=str))
    
    # Initialize Terraform automation
    tf_automation = TerraformAutomation("your-project-id")
    
    # Generate and apply optimizations (dry run)
    sample_recommendations = [
        {
            'resource_name': 'instance-1',
            'recommended_type': 'n2-standard-2 (CUD)',
            'monthly_savings': 150
        }
    ]
    
    tf_results = tf_automation.apply_optimizations(sample_recommendations, dry_run=True)
    print(f"\nTerraform Results: {tf_results}")