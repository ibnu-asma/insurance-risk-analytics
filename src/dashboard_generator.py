import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class RiskPricingDashboard:
    """Dashboard generator for risk-based pricing analysis."""
    
    def __init__(self, df: pd.DataFrame, risk_adjusted_df: pd.DataFrame = None):
        self.df = df
        self.risk_adjusted_df = risk_adjusted_df or df
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_comprehensive_dashboard(self, save_path: str = None) -> None:
        """Generate comprehensive dashboard with all key metrics."""
        fig = plt.figure(figsize=(20, 24))
        
        # Create grid layout
        gs = fig.add_gridspec(6, 4, hspace=0.3, wspace=0.3)
        
        # 1. Risk Factor Summary
        self._plot_risk_factors(fig.add_subplot(gs[0, :2]))
        
        # 2. Premium Distribution Comparison
        self._plot_premium_comparison(fig.add_subplot(gs[0, 2:]))
        
        # 3. Province Risk Analysis
        self._plot_province_analysis(fig.add_subplot(gs[1, :2]))
        
        # 4. Gender Risk Analysis
        self._plot_gender_analysis(fig.add_subplot(gs[1, 2:]))
        
        # 5. Adjustment Distribution
        self._plot_adjustment_distribution(fig.add_subplot(gs[2, :2]))
        
        # 6. Financial Impact
        self._plot_financial_impact(fig.add_subplot(gs[2, 2:]))
        
        # 7. Risk vs Reward Scatter
        self._plot_risk_reward_scatter(fig.add_subplot(gs[3, :2]))
        
        # 8. Compliance Status
        self._plot_compliance_status(fig.add_subplot(gs[3, 2:]))
        
        # 9. Portfolio Performance
        self._plot_portfolio_performance(fig.add_subplot(gs[4, :]))
        
        # 10. Implementation Timeline
        self._plot_implementation_timeline(fig.add_subplot(gs[5, :]))
        
        plt.suptitle('Risk-Based Pricing Implementation Dashboard', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_risk_factors(self, ax):
        """Plot risk factors summary."""
        # Example risk factors (would come from actual implementation)
        risk_factors = {
            'Gauteng': 1.15,
            'Western Cape': 0.95,
            'KwaZulu-Natal': 1.10,
            'Male': 1.12,
            'Female': 0.98
        }
        
        categories = list(risk_factors.keys())
        values = list(risk_factors.values())
        colors = ['red' if v > 1 else 'green' for v in values]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Risk Adjustment Factors')
        ax.set_ylabel('Risk Factor')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_premium_comparison(self, ax):
        """Plot premium distribution comparison."""
        if 'RiskAdjustedPremium' in self.risk_adjusted_df.columns:
            sample_data = self.risk_adjusted_df.sample(n=min(1000, len(self.risk_adjusted_df)))
            
            ax.hist(sample_data['TotalPremium'], bins=30, alpha=0.6, label='Base Premium', color='blue')
            ax.hist(sample_data['RiskAdjustedPremium'], bins=30, alpha=0.6, label='Risk-Adjusted Premium', color='orange')
            ax.set_title('Premium Distribution Comparison')
            ax.set_xlabel('Premium Amount (R)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Risk-adjusted premiums not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Premium Distribution')
    
    def _plot_province_analysis(self, ax):
        """Plot province risk analysis."""
        if 'Province' in self.df.columns:
            province_stats = self.df.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'HasClaim': 'mean'
            })
            
            # Calculate loss ratio
            province_stats['LossRatio'] = province_stats['TotalClaims'] / province_stats['TotalPremium']
            
            # Plot top provinces by loss ratio
            top_provinces = province_stats['LossRatio'].sort_values(ascending=False).head(5)
            
            bars = ax.bar(range(len(top_provinces)), top_provinces.values, color='coral')
            ax.set_title('Loss Ratio by Province (Top 5)')
            ax.set_ylabel('Loss Ratio')
            ax.set_xticks(range(len(top_provinces)))
            ax.set_xticklabels(top_provinces.index, rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, top_provinces.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Province data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Province Risk Analysis')
    
    def _plot_gender_analysis(self, ax):
        """Plot gender risk analysis."""
        if 'Gender' in self.df.columns:
            gender_stats = self.df.groupby('Gender').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'HasClaim': 'mean'
            })
            
            # Calculate loss ratio
            gender_stats['LossRatio'] = gender_stats['TotalClaims'] / gender_stats['TotalPremium']
            
            bars = ax.bar(gender_stats.index, gender_stats['LossRatio'], color=['pink', 'lightblue'])
            ax.set_title('Loss Ratio by Gender')
            ax.set_ylabel('Loss Ratio')
            
            # Add value labels
            for bar, value in zip(bars, gender_stats['LossRatio']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Gender data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Gender Risk Analysis')
    
    def _plot_adjustment_distribution(self, ax):
        """Plot adjustment distribution."""
        if 'RiskAdjustmentPercent' in self.risk_adjusted_df.columns:
            ax.hist(self.risk_adjusted_df['RiskAdjustmentPercent'], bins=50, alpha=0.7, color='green')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax.set_title('Distribution of Premium Adjustments')
            ax.set_xlabel('Adjustment (%)')
            ax.set_ylabel('Frequency')
        else:
            ax.text(0.5, 0.5, 'Adjustment data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Adjustment Distribution')
    
    def _plot_financial_impact(self, ax):
        """Plot financial impact analysis."""
        if 'RiskAdjustedPremium' in self.risk_adjusted_df.columns:
            total_base = self.risk_adjusted_df['TotalPremium'].sum()
            total_adjusted = self.risk_adjusted_df['RiskAdjustedPremium'].sum()
            impact = total_adjusted - total_base
            
            categories = ['Base Premium', 'Risk-Adjusted Premium']
            values = [total_base, total_adjusted]
            colors = ['lightblue', 'lightgreen']
            
            bars = ax.bar(categories, values, color=colors)
            ax.set_title('Financial Impact (Total Portfolio)')
            ax.set_ylabel('Amount (R)')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_base * 0.01,
                       f'R{value:,.0f}', ha='center', va='bottom')
            
            # Add impact annotation
            ax.text(0.5, 0.8, f'Impact: R{impact:+,.0f}\n({(impact/total_base)*100:+.1f}%)',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Financial impact data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Financial Impact')
    
    def _plot_risk_reward_scatter(self, ax):
        """Plot risk vs reward scatter plot."""
        if 'RiskAdjustmentPercent' in self.risk_adjusted_df.columns and 'TotalClaims' in self.risk_adjusted_df.columns:
            sample_data = self.risk_adjusted_df.sample(n=min(1000, len(self.risk_adjusted_df)))
            
            scatter = ax.scatter(sample_data['RiskAdjustmentPercent'], sample_data['TotalClaims'], 
                               alpha=0.6, c=sample_data['TotalPremium'], cmap='viridis')
            ax.set_title('Risk vs Claims (Premium-Adjusted)')
            ax.set_xlabel('Risk Adjustment (%)')
            ax.set_ylabel('Total Claims (R)')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Total Premium (R)')
        else:
            ax.text(0.5, 0.5, 'Risk-reward data not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Risk vs Reward Analysis')
    
    def _plot_compliance_status(self, ax):
        """Plot compliance status."""
        # Example compliance status (would come from actual compliance checker)
        compliance_data = {
            'Province Pricing': 'COMPLIANT',
            'Gender Pricing': 'REVIEW_REQUIRED',
            'Documentation': 'PENDING',
            'Regulatory Review': 'IN_PROGRESS'
        }
        
        status_colors = {
            'COMPLIANT': 'green',
            'REVIEW_REQUIRED': 'orange',
            'PENDING': 'yellow',
            'IN_PROGRESS': 'blue',
            'NON_COMPLIANT': 'red'
        }
        
        categories = list(compliance_data.keys())
        statuses = list(compliance_data.values())
        colors = [status_colors.get(status, 'gray') for status in statuses]
        
        bars = ax.bar(categories, [1] * len(categories), color=colors, alpha=0.7)
        ax.set_title('Regulatory Compliance Status')
        ax.set_ylabel('Status')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45)
        ax.set_yticks([])
        
        # Add status labels
        for bar, status in zip(bars, statuses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                   status, ha='center', va='center', fontweight='bold')
    
    def _plot_portfolio_performance(self, ax):
        """Plot portfolio performance metrics."""
        # Calculate key performance indicators
        total_premium = self.df['TotalPremium'].sum()
        total_claims = self.df['TotalClaims'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        claim_frequency = self.df['HasClaim'].mean()
        
        # Create performance summary
        metrics = ['Loss Ratio', 'Claim Frequency', 'Total Premium', 'Total Claims']
        values = [loss_ratio, claim_frequency, total_premium/1e6, total_claims/1e6]
        units = ['', '', '(R millions)', '(R millions)']
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, color=['red', 'orange', 'green', 'blue'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel('Value')
        ax.set_title('Portfolio Performance Summary')
        
        # Add value labels
        for bar, value, unit in zip(bars, values, units):
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f} {unit}', ha='left', va='center')
    
    def _plot_implementation_timeline(self, ax):
        """Plot implementation timeline."""
        # Implementation phases
        phases = ['Planning', 'Compliance Review', 'System Updates', 'Testing', 'Deployment', 'Monitoring']
        durations = [30, 45, 60, 30, 15, 90]  # days
        colors = ['lightblue', 'orange', 'green', 'yellow', 'red', 'purple']
        
        # Create Gantt-style timeline
        y_pos = np.arange(len(phases))
        bars = ax.barh(y_pos, durations, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(phases)
        ax.set_xlabel('Duration (Days)')
        ax.set_title('Implementation Timeline')
        
        # Add duration labels
        for bar, duration in zip(bars, durations):
            ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                   f'{duration} days', ha='center', va='center', fontweight='bold')
    
    def generate_summary_report(self) -> str:
        """Generate text summary report."""
        report = []
        report.append("=== RISK-BASED PRICING IMPLEMENTATION SUMMARY ===\n")
        
        # Key statistics
        total_records = len(self.df)
        total_premium = self.df['TotalPremium'].sum()
        total_claims = self.df['TotalClaims'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        report.append(f"PORTFOLIO OVERVIEW:")
        report.append(f"  • Total Records: {total_records:,}")
        report.append(f"  • Total Premium: R{total_premium:,.2f}")
        report.append(f"  • Total Claims: R{total_claims:,.2f}")
        report.append(f"  • Loss Ratio: {loss_ratio:.3f}")
        
        if 'RiskAdjustedPremium' in self.risk_adjusted_df.columns:
            total_adjusted = self.risk_adjusted_df['RiskAdjustedPremium'].sum()
            impact = total_adjusted - total_premium
            impact_pct = (impact / total_premium) * 100
            
            report.append(f"\nRISK ADJUSTMENT IMPACT:")
            report.append(f"  • Risk-Adjusted Premium: R{total_adjusted:,.2f}")
            report.append(f"  • Total Impact: R{impact:+,.2f} ({impact_pct:+.2f}%)")
        
        report.append(f"\nIMPLEMENTATION STATUS:")
        report.append(f"  • Risk Factors: Defined and validated")
        report.append(f"  • Pricing Engine: Implemented")
        report.append(f"  • Compliance Review: In progress")
        report.append(f"  • System Integration: Pending")
        
        report.append(f"\nNEXT STEPS:")
        report.append(f"  1. Complete regulatory compliance review")
        report.append(f"  2. Update pricing systems")
        report.append(f"  3. Train underwriting team")
        report.append(f"  4. Monitor performance")
        
        return "\n".join(report)

def create_dashboard_from_data(data_path: str, save_path: str = None) -> None:
    """Create dashboard from data file."""
    try:
        df = pd.read_csv(data_path)
        dashboard = RiskPricingDashboard(df)
        dashboard.generate_comprehensive_dashboard(save_path)
        
        # Print summary report
        print(dashboard.generate_summary_report())
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")

if __name__ == '__main__':
    # Example usage
    print("Risk Pricing Dashboard Generator")
    print("Use create_dashboard_from_data() to generate dashboard from CSV file")

    # Create comprehensive visualizations 