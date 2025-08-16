import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskAdjustment:
    """Risk adjustment factors based on hypothesis testing results."""
    province_adjustments: Dict[str, float]
    gender_adjustments: Dict[str, float]
    base_premium_multiplier: float = 1.0

class RiskPricingEngine:
    """Risk-based pricing engine using hypothesis testing results."""
    
    def __init__(self):
        # Based on hypothesis testing: Provinces show significant differences
        # Gauteng has higher risk than Western Cape
        self.province_risk_factors = {
            'Gauteng': 1.15,  # 15% premium increase for higher risk
            'Western Cape': 0.95,  # 5% premium decrease for lower risk
            'KwaZulu-Natal': 1.10,  # 10% increase
            'Free State': 1.05,  # 5% increase
            'Mpumalanga': 1.08,  # 8% increase
            'Limpopo': 1.12,  # 12% increase
            'North West': 1.06,  # 6% increase
            'Northern Cape': 1.03,  # 3% increase
        }
        
        # Based on hypothesis testing: Gender shows significant differences
        # Note: Requires regulatory compliance review
        self.gender_risk_factors = {
            'Male': 1.12,  # 12% premium increase
            'Female': 0.98,  # 2% premium decrease
        }
        
        self.base_premium_multiplier = 1.0
    
    def calculate_risk_adjusted_premium(self, 
                                      base_premium: float,
                                      province: str,
                                      gender: str,
                                      sum_insured: float,
                                      vehicle_type: str = None) -> Dict[str, float]:
        """
        Calculate risk-adjusted premium based on hypothesis testing results.
        
        Args:
            base_premium: Base premium amount
            province: Province name
            gender: Gender (Male/Female)
            sum_insured: Sum insured amount
            vehicle_type: Optional vehicle type for additional adjustments
            
        Returns:
            Dictionary with premium breakdown and adjustments
        """
        # Get risk factors
        province_factor = self.province_risk_factors.get(province, 1.0)
        gender_factor = self.gender_risk_factors.get(gender, 1.0)
        
        # Calculate adjustments
        province_adjustment = base_premium * (province_factor - 1.0)
        gender_adjustment = base_premium * (gender_factor - 1.0)
        
        # Apply risk factors
        risk_adjusted_premium = base_premium * province_factor * gender_factor
        
        # Additional adjustments based on sum insured (if needed)
        sum_insured_factor = self._calculate_sum_insured_factor(sum_insured)
        final_premium = risk_adjusted_premium * sum_insured_factor
        
        return {
            'base_premium': base_premium,
            'province_adjustment': province_adjustment,
            'gender_adjustment': gender_adjustment,
            'province_factor': province_factor,
            'gender_factor': gender_factor,
            'sum_insured_factor': sum_insured_factor,
            'risk_adjusted_premium': risk_adjusted_premium,
            'final_premium': final_premium,
            'total_adjustment_percent': ((final_premium / base_premium) - 1.0) * 100
        }
    
    def _calculate_sum_insured_factor(self, sum_insured: float) -> float:
        """Calculate adjustment factor based on sum insured."""
        if sum_insured <= 100000:
            return 1.0
        elif sum_insured <= 500000:
            return 1.02
        elif sum_insured <= 1000000:
            return 1.05
        else:
            return 1.08
    
    def get_risk_report(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate risk analysis report based on hypothesis testing results.
        
        Args:
            df: DataFrame with insurance data
            
        Returns:
            Risk analysis report
        """
        # Province risk analysis
        province_stats = df.groupby('Province').agg({
            'TotalClaims': ['sum', 'mean', 'count'],
            'TotalPremium': ['sum', 'mean'],
            'HasClaim': 'mean'
        }).round(4)
        
        # Gender risk analysis
        gender_stats = df.groupby('Gender').agg({
            'TotalClaims': ['sum', 'mean', 'count'],
            'TotalPremium': ['sum', 'mean'],
            'HasClaim': 'mean'
        }).round(4)
        
        # Calculate loss ratios
        province_stats['LossRatio'] = (province_stats[('TotalClaims', 'sum')] / 
                                      province_stats[('TotalPremium', 'sum')]).round(4)
        gender_stats['LossRatio'] = (gender_stats[('TotalClaims', 'sum')] / 
                                    gender_stats[('TotalPremium', 'sum')]).round(4)
        
        return {
            'province_risk_analysis': province_stats,
            'gender_risk_analysis': gender_stats,
            'recommended_adjustments': {
                'high_risk_provinces': ['Gauteng', 'KwaZulu-Natal', 'Limpopo'],
                'low_risk_provinces': ['Western Cape', 'Northern Cape'],
                'gender_considerations': 'Review regulatory compliance for gender-based pricing'
            }
        }

def apply_risk_pricing_to_dataset(df: pd.DataFrame, 
                                 base_premium_col: str = 'TotalPremium',
                                 output_col: str = 'RiskAdjustedPremium') -> pd.DataFrame:
    """
    Apply risk-based pricing to entire dataset.
    
    Args:
        df: Input DataFrame
        base_premium_col: Column name for base premium
        output_col: Column name for risk-adjusted premium
        
    Returns:
        DataFrame with risk-adjusted premiums
    """
    pricing_engine = RiskPricingEngine()
    
    # Apply pricing adjustments
    risk_adjustments = []
    for idx, row in df.iterrows():
        adjustment = pricing_engine.calculate_risk_adjusted_premium(
            base_premium=row[base_premium_col],
            province=row['Province'],
            gender=row['Gender'],
            sum_insured=row.get('SumInsured', 0)
        )
        risk_adjustments.append(adjustment['final_premium'])
    
    df[output_col] = risk_adjustments
    df['RiskAdjustmentPercent'] = ((df[output_col] / df[base_premium_col]) - 1.0) * 100
    
    return df

if __name__ == '__main__':
    # Example usage
    pricing_engine = RiskPricingEngine()
    
    # Test with sample data
    sample_premium = 1000
    sample_adjustment = pricing_engine.calculate_risk_adjusted_premium(
        base_premium=sample_premium,
        province='Gauteng',
        gender='Male',
        sum_insured=500000
    )
    
    print("Sample Risk Adjustment:")
    for key, value in sample_adjustment.items():
        print(f"{key}: {value}") 