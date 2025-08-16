import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class ComplianceChecker:
    """Regulatory compliance checker for risk-based pricing."""
    
    def __init__(self):
        # Regulatory guidelines (example - should be updated with actual regulations)
        self.regulatory_guidelines = {
            'gender_pricing': {
                'allowed': False,  # Many jurisdictions prohibit gender-based pricing
                'exceptions': ['Actuarial justification required'],
                'documentation_required': True,
                'review_frequency': 'Annual'
            },
            'province_pricing': {
                'allowed': True,
                'documentation_required': True,
                'review_frequency': 'Annual'
            },
            'zip_code_pricing': {
                'allowed': True,
                'documentation_required': True,
                'review_frequency': 'Annual'
            }
        }
        
        self.compliance_history = []
    
    def check_gender_pricing_compliance(self, 
                                      risk_factors: Dict[str, float],
                                      justification_data: Dict[str, any]) -> Dict[str, any]:
        """
        Check compliance for gender-based pricing.
        
        Args:
            risk_factors: Gender-based risk adjustment factors
            justification_data: Statistical justification data
            
        Returns:
            Compliance assessment
        """
        compliance_report = {
            'compliant': False,
            'warnings': [],
            'recommendations': [],
            'required_actions': [],
            'risk_level': 'HIGH'
        }
        
        # Check if gender pricing is allowed
        if not self.regulatory_guidelines['gender_pricing']['allowed']:
            compliance_report['warnings'].append(
                "Gender-based pricing is prohibited in this jurisdiction"
            )
            compliance_report['required_actions'].append(
                "Remove gender-based pricing adjustments"
            )
            return compliance_report
        
        # Check statistical justification
        if not self._validate_statistical_justification(justification_data):
            compliance_report['warnings'].append(
                "Insufficient statistical justification for gender-based pricing"
            )
            compliance_report['required_actions'].append(
                "Provide additional actuarial justification"
            )
        
        # Check for discrimination
        if self._check_discrimination_risk(risk_factors):
            compliance_report['warnings'].append(
                "Potential discrimination risk in gender-based pricing"
            )
            compliance_report['required_actions'].append(
                "Review pricing factors for fairness and non-discrimination"
            )
        
        # If no major issues, mark as compliant
        if not compliance_report['warnings']:
            compliance_report['compliant'] = True
            compliance_report['risk_level'] = 'LOW'
        
        return compliance_report
    
    def _validate_statistical_justification(self, justification_data: Dict[str, any]) -> bool:
        """Validate statistical justification for gender-based pricing."""
        required_fields = ['p_value', 'sample_size', 'effect_size', 'confidence_interval']
        
        for field in required_fields:
            if field not in justification_data:
                return False
        
        # Check if p-value is significant
        if justification_data.get('p_value', 1.0) > 0.05:
            return False
        
        # Check if sample size is adequate
        if justification_data.get('sample_size', 0) < 100:
            return False
        
        return True
    
    def _check_discrimination_risk(self, risk_factors: Dict[str, float]) -> bool:
        """Check for potential discrimination in risk factors."""
        if 'Male' in risk_factors and 'Female' in risk_factors:
            ratio = risk_factors['Male'] / risk_factors['Female']
            # Flag if ratio is too extreme (e.g., > 1.5 or < 0.67)
            if ratio > 1.5 or ratio < 0.67:
                return True
        return False
    
    def generate_compliance_report(self, 
                                 pricing_engine_results: Dict[str, any],
                                 hypothesis_testing_results: Dict[str, any]) -> Dict[str, any]:
        """
        Generate comprehensive compliance report.
        
        Args:
            pricing_engine_results: Results from risk pricing engine
            hypothesis_testing_results: Results from hypothesis testing
            
        Returns:
            Comprehensive compliance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_compliance': 'PENDING_REVIEW',
            'sections': {},
            'recommendations': [],
            'required_actions': []
        }
        
        # Check gender pricing compliance
        gender_justification = {
            'p_value': hypothesis_testing_results.get('Gender', {}).get('Claim Frequency', {}).get('p', 1.0),
            'sample_size': hypothesis_testing_results.get('Gender', {}).get('sample_size', 0),
            'effect_size': 'medium',  # Would need to calculate actual effect size
            'confidence_interval': [0.95, 1.05]  # Example
        }
        
        gender_compliance = self.check_gender_pricing_compliance(
            pricing_engine_results.get('gender_risk_factors', {}),
            gender_justification
        )
        
        report['sections']['gender_pricing'] = gender_compliance
        
        # Check province pricing compliance
        province_compliance = {
            'compliant': True,
            'warnings': [],
            'recommendations': ['Monitor for geographic discrimination'],
            'required_actions': ['Document province-based risk factors']
        }
        report['sections']['province_pricing'] = province_compliance
        
        # Overall assessment
        if all(section.get('compliant', False) for section in report['sections'].values()):
            report['overall_compliance'] = 'COMPLIANT'
        elif any(section.get('compliant', False) for section in report['sections'].values()):
            report['overall_compliance'] = 'PARTIALLY_COMPLIANT'
        else:
            report['overall_compliance'] = 'NON_COMPLIANT'
        
        # Aggregate recommendations and actions
        for section in report['sections'].values():
            report['recommendations'].extend(section.get('recommendations', []))
            report['required_actions'].extend(section.get('required_actions', []))
        
        # Remove duplicates
        report['recommendations'] = list(set(report['recommendations']))
        report['required_actions'] = list(set(report['required_actions']))
        
        return report
    
    def save_compliance_report(self, report: Dict[str, any], filename: str = None):
        """Save compliance report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"compliance_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Compliance report saved to {filename}")

def create_compliance_summary(hypothesis_results: Dict[str, any]) -> str:
    """
    Create a human-readable compliance summary.
    
    Args:
        hypothesis_results: Results from hypothesis testing
        
    Returns:
        Compliance summary text
    """
    summary = []
    summary.append("=== REGULATORY COMPLIANCE SUMMARY ===\n")
    
    # Gender pricing assessment
    gender_p_value = hypothesis_results.get('Gender', {}).get('Claim Frequency', {}).get('p', 1.0)
    if gender_p_value < 0.05:
        summary.append("⚠️  GENDER-BASED PRICING ALERT:")
        summary.append("   - Statistically significant gender differences detected")
        summary.append("   - p-value: {:.4f}".format(gender_p_value))
        summary.append("   - REQUIRED: Regulatory compliance review")
        summary.append("   - RECOMMENDATION: Consult legal/compliance team\n")
    else:
        summary.append("✅ GENDER-BASED PRICING:")
        summary.append("   - No significant gender differences")
        summary.append("   - Gender-based pricing not recommended\n")
    
    # Province pricing assessment
    province_p_value = hypothesis_results.get('Provinces', {}).get('Claim Severity', {}).get('p', 1.0)
    if province_p_value < 0.05:
        summary.append("✅ PROVINCE-BASED PRICING:")
        summary.append("   - Statistically significant provincial differences")
        summary.append("   - p-value: {:.4f}".format(province_p_value))
        summary.append("   - RECOMMENDATION: Implement province-based adjustments\n")
    
    summary.append("=== NEXT STEPS ===")
    summary.append("1. Review gender pricing with compliance team")
    summary.append("2. Document province-based risk factors")
    summary.append("3. Implement approved pricing adjustments")
    summary.append("4. Monitor for regulatory changes")
    
    return "\n".join(summary)

if __name__ == '__main__':
    # Example usage
    checker = ComplianceChecker()
    
    # Sample hypothesis testing results
    sample_results = {
        'Gender': {
            'Claim Frequency': {'p': 0.0198},
            'sample_size': 2788
        },
        'Provinces': {
            'Claim Severity': {'p': 0.0306}
        }
    }
    
    # Generate compliance summary
    summary = create_compliance_summary(sample_results)
    print(summary) 