import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import warnings
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os

warnings.filterwarnings('ignore')

class InsurancePredictiveModels:
    """Comprehensive predictive modeling for insurance risk-based pricing."""
    
    def __init__(self, data_path: str = None, df: pd.DataFrame = None):
        """
        Initialize the predictive modeling system.
        
        Args:
            data_path: Path to the CSV file
            df: Pre-loaded DataFrame
        """
        self.df = df if df is not None else pd.read_csv(data_path) if data_path else None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        self.shap_values = {}
        self.model_performance = {}
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Processed data
        self.X_processed = None
        self.y_claim_severity = None
        self.y_claim_probability = None
        self.y_premium = None
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Prepare data for modeling including feature engineering and preprocessing.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        print("=== DATA PREPARATION ===")
        
        # Create copy for processing
        df = self.df.copy()
        
        # 1. Feature Engineering
        print("1. Feature Engineering...")
        df = self._engineer_features(df)
        
        # 2. Handle Missing Data
        print("2. Handling Missing Data...")
        df = self._handle_missing_data(df)
        
        # 3. Encode Categorical Data
        print("3. Encoding Categorical Data...")
        df = self._encode_categorical_data(df)
        
        # 4. Create Target Variables
        print("4. Creating Target Variables...")
        self._create_target_variables(df)
        
        # 5. Feature Selection
        print("5. Feature Selection...")
        self.X_processed = self._select_features(df)
        
        # 6. Train-Test Split
        print("6. Train-Test Split...")
        self._split_data(test_size, random_state)
        
        print(f"Data preparation complete. Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features for modeling."""
        
        # Time-based features
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            df['TransactionYear'] = df['TransactionMonth'].dt.year
            df['TransactionMonthNum'] = df['TransactionMonth'].dt.month
            df['TransactionQuarter'] = df['TransactionMonth'].dt.quarter
        
        # Vehicle age
        if 'RegistrationYear' in df.columns:
            df['VehicleAge'] = 2024 - df['RegistrationYear']  # Assuming current year
            df['VehicleAge'] = df['VehicleAge'].clip(lower=0, upper=50)
        
        # Risk indicators
        df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        df['ClaimFrequency'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
        
        # Premium adequacy
        df['PremiumAdequacy'] = df['TotalPremium'] / df['SumInsured'].replace(0, 1)
        
        # Vehicle value indicators
        if 'CustomValueEstimate' in df.columns:
            df['ValueToSumInsuredRatio'] = df['CustomValueEstimate'] / df['SumInsured'].replace(0, 1)
        
        # Risk zones (if available)
        if 'MainCrestaZone' in df.columns:
            df['HighRiskZone'] = df['MainCrestaZone'].isin(['A', 'B', 'C']).astype(int)
        
        # Cross-border risk
        if 'CrossBorder' in df.columns:
            df['CrossBorderRisk'] = (df['CrossBorder'] == 'Yes').astype(int)
        
        # Fleet size indicator
        if 'NumberOfVehiclesInFleet' in df.columns:
            df['IsFleet'] = (df['NumberOfVehiclesInFleet'] > 1).astype(int)
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        
        # Numerical columns - impute with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Categorical columns - impute with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        
        return df
    
    def _encode_categorical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 50:  # Only encode if reasonable number of categories
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> None:
        """Create target variables for different modeling tasks."""
        
        # 1. Claim Severity (for policies with claims)
        self.y_claim_severity = df[df['TotalClaims'] > 0]['TotalClaims'].copy()
        
        # 2. Claim Probability (binary)
        self.y_claim_probability = df['HasClaim'].copy()
        
        # 3. Premium (for premium optimization)
        self.y_premium = df['TotalPremium'].copy()
        
        print(f"Target variables created:")
        print(f"  - Claim Severity: {len(self.y_claim_severity)} samples")
        print(f"  - Claim Probability: {len(self.y_claim_probability)} samples")
        print(f"  - Premium: {len(self.y_premium)} samples")
    
    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for modeling."""
        
        # Remove target variables and irrelevant columns
        exclude_cols = [
            'TotalClaims', 'TotalPremium', 'HasClaim', 'ClaimFrequency', 'LossRatio',
            'TransactionMonth', 'CalculatedPremiumPerTerm'  # Remove if using as target
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select features
        X = df[feature_cols].copy()
        
        # Remove columns with too many unique values (likely IDs)
        for col in X.columns:
            if X[col].nunique() > len(X) * 0.5:  # More than 50% unique values
                X = X.drop(columns=[col])
        
        print(f"Selected {len(X.columns)} features for modeling")
        return X
    
    def _split_data(self, test_size: float, random_state: int) -> None:
        """Split data into training and testing sets."""
        
        # For claim severity (only policies with claims)
        if len(self.y_claim_severity) > 0:
            X_severity = self.X_processed.loc[self.y_claim_severity.index]
            self.X_train_severity, self.X_test_severity, self.y_train_severity, self.y_test_severity = train_test_split(
                X_severity, self.y_claim_severity, test_size=test_size, random_state=random_state
            )
        
        # For claim probability and premium (all policies)
        self.X_train, self.X_test, self.y_train_prob, self.y_test_prob = train_test_split(
            self.X_processed, self.y_claim_probability, test_size=test_size, random_state=random_state
        )
        
        _, _, self.y_train_premium, self.y_test_premium = train_test_split(
            self.X_processed, self.y_premium, test_size=test_size, random_state=random_state
        )
    
    def build_claim_severity_model(self) -> Dict[str, Any]:
        """
        Build models to predict claim severity (TotalClaims for policies with claims).
        
        Returns:
            Dictionary with model performance metrics
        """
        print("\n=== CLAIM SEVERITY PREDICTION ===")
        
        if len(self.y_claim_severity) == 0:
            print("No claims data available for severity modeling")
            return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train_severity)
        X_test_scaled = self.scaler.transform(self.X_test_severity)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, self.y_train_severity)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test_severity, y_pred))
            r2 = r2_score(self.y_test_severity, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: R{rmse:,.2f}")
            print(f"  R²: {r2:.4f}")
            
            # Store best model
            if name not in self.models or r2 > self.model_performance.get(name, {}).get('r2', 0):
                self.models[f'severity_{name}'] = model
                self.model_performance[f'severity_{name}'] = {'rmse': rmse, 'r2': r2}
        
        return results
    
    def build_claim_probability_model(self) -> Dict[str, Any]:
        """
        Build models to predict claim probability (binary classification).
        
        Returns:
            Dictionary with model performance metrics
        """
        print("\n=== CLAIM PROBABILITY PREDICTION ===")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, self.y_train_prob)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test_prob, y_pred)
            precision = precision_score(self.y_test_prob, y_pred)
            recall = recall_score(self.y_test_prob, y_pred)
            f1 = f1_score(self.y_test_prob, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            
            # Store best model
            if name not in self.models or f1 > self.model_performance.get(name, {}).get('f1', 0):
                self.models[f'probability_{name}'] = model
                self.model_performance[f'probability_{name}'] = {
                    'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
                }
        
        return results
    
    def build_premium_optimization_model(self) -> Dict[str, Any]:
        """
        Build models to predict optimal premium amounts.
        
        Returns:
            Dictionary with model performance metrics
        """
        print("\n=== PREMIUM OPTIMIZATION ===")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, self.y_train_premium)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test_premium, y_pred))
            r2 = r2_score(self.y_test_premium, y_pred)
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: R{rmse:,.2f}")
            print(f"  R²: {r2:.4f}")
            
            # Store best model
            if name not in self.models or r2 > self.model_performance.get(name, {}).get('r2', 0):
                self.models[f'premium_{name}'] = model
                self.model_performance[f'premium_{name}'] = {'rmse': rmse, 'r2': r2}
        
        return results
    
    def analyze_feature_importance(self, model_name: str = None) -> Dict[str, Any]:
        """
        Analyze feature importance using SHAP values.
        
        Args:
            model_name: Name of the model to analyze (if None, analyze best model)
            
        Returns:
            Dictionary with feature importance analysis
        """
        print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if model_name is None:
            # Find best model based on performance
            best_model = max(self.model_performance.items(), key=lambda x: x[1].get('r2', x[1].get('f1', 0)))
            model_name = best_model[0]
        
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return {}
        
        model = self.models[model_name]
        
        # Prepare data for SHAP analysis
        if 'severity' in model_name:
            X_sample = self.X_test_severity.sample(n=min(1000, len(self.X_test_severity)))
        else:
            X_sample = self.X_test.sample(n=min(1000, len(self.X_test)))
        
        X_sample_scaled = self.scaler.transform(X_sample)
        
        # Calculate SHAP values
        if hasattr(model, 'predict_proba'):
            # Classification model
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_sample_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
        else:
            # Regression model
            explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.LinearExplainer(model, X_sample_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)
        
        # Get feature importance
        feature_importance = np.abs(shap_values).mean(0)
        feature_names = X_sample.columns
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Store results
        self.feature_importance[model_name] = importance_df
        self.shap_values[model_name] = shap_values
        
        print(f"Top 10 Most Important Features for {model_name}:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return {
            'importance_df': importance_df,
            'shap_values': shap_values,
            'feature_names': feature_names
        }
    
    def generate_risk_based_premium(self, 
                                  claim_prob_model: str = None,
                                  claim_severity_model: str = None,
                                  expense_loading: float = 0.15,
                                  profit_margin: float = 0.10) -> pd.Series:
        """
        Generate risk-based premium using the formula:
        Premium = (Predicted Probability of Claim * Predicted Claim Severity) + Expense Loading + Profit Margin
        
        Args:
            claim_prob_model: Name of the claim probability model
            claim_severity_model: Name of the claim severity model
            expense_loading: Expense loading factor (default 15%)
            profit_margin: Profit margin factor (default 10%)
            
        Returns:
            Series with risk-based premiums
        """
        print(f"\n=== RISK-BASED PREMIUM GENERATION ===")
        
        # Use best models if not specified
        if claim_prob_model is None:
            claim_prob_model = max([k for k in self.models.keys() if 'probability' in k], 
                                 key=lambda x: self.model_performance[x].get('f1', 0))
        
        if claim_severity_model is None:
            claim_severity_model = max([k for k in self.models.keys() if 'severity' in k], 
                                     key=lambda x: self.model_performance[x].get('r2', 0))
        
        # Get models
        prob_model = self.models[claim_prob_model]
        severity_model = self.models[claim_severity_model]
        
        # Prepare test data
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Predict claim probability
        claim_prob = prob_model.predict_proba(X_test_scaled)[:, 1]
        
        # Predict claim severity (use average for policies without claims)
        claim_severity = severity_model.predict(X_test_scaled)
        
        # Calculate risk-based premium
        risk_premium = claim_prob * claim_severity
        total_premium = risk_premium * (1 + expense_loading + profit_margin)
        
        print(f"Risk-based premium calculation complete:")
        print(f"  - Average claim probability: {claim_prob.mean():.4f}")
        print(f"  - Average claim severity: R{claim_severity.mean():,.2f}")
        print(f"  - Average risk premium: R{risk_premium.mean():,.2f}")
        print(f"  - Average total premium: R{total_premium.mean():,.2f}")
        
        return pd.Series(total_premium, index=self.X_test.index)
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """
        Evaluate all models and generate comprehensive comparison.
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        print("\n=== COMPREHENSIVE MODEL EVALUATION ===")
        
        # Build all models
        severity_results = self.build_claim_severity_model()
        probability_results = self.build_claim_probability_model()
        premium_results = self.build_premium_optimization_model()
        
        # Analyze feature importance for best models
        self.analyze_feature_importance()
        
        # Generate risk-based premium
        risk_premium = self.generate_risk_based_premium()
        
        # Compile results
        evaluation_results = {
            'claim_severity': severity_results,
            'claim_probability': probability_results,
            'premium_optimization': premium_results,
            'risk_based_premium': risk_premium,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance
        }
        
        return evaluation_results
    
    def save_models(self, directory: str = 'models') -> None:
        """Save trained models to disk."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name, model in self.models.items():
            filename = os.path.join(directory, f'{name}.joblib')
            joblib.dump(model, filename)
            print(f"Saved model: {filename}")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(directory, 'label_encoders.joblib'))
        print("Saved preprocessing components")
    
    def load_models(self, directory: str = 'models') -> None:
        """Load trained models from disk."""
        for filename in os.listdir(directory):
            if filename.endswith('.joblib') and not filename.startswith('scaler') and not filename.startswith('label_encoders'):
                model_name = filename.replace('.joblib', '')
                model_path = os.path.join(directory, filename)
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded model: {model_name}")
        
        # Load preprocessing components
        self.scaler = joblib.load(os.path.join(directory, 'scaler.joblib'))
        self.label_encoders = joblib.load(os.path.join(directory, 'label_encoders.joblib'))

def create_modeling_pipeline(data_path: str = None, df: pd.DataFrame = None) -> InsurancePredictiveModels:
    """
    Create and run the complete modeling pipeline.
    
    Args:
        data_path: Path to the data file
        df: Pre-loaded DataFrame
        
    Returns:
        Trained InsurancePredictiveModels instance
    """
    # Initialize modeling system
    modeler = InsurancePredictiveModels(data_path=data_path, df=df)
    
    # Prepare data
    modeler.prepare_data()
    
    # Evaluate all models
    results = modeler.evaluate_all_models()
    
    return modeler, results