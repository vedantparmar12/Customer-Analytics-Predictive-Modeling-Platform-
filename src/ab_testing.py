import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower, tt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from .logger import get_logger
from .custom_exception import CustomException

logger = get_logger(__name__)

class ABTestingFramework:
    def __init__(self):
        self.test_results = {}
        self.power_analysis = TTestPower()
        logger.info("A/B Testing Framework initialized")
    
    def calculate_sample_size(self, baseline_rate, minimum_detectable_effect, 
                            alpha=0.05, power=0.8, test_type='conversion'):
        """Calculate required sample size for A/B test"""
        try:
            logger.info(f"Calculating sample size for {test_type} test...")
            
            if test_type == 'conversion':
                # For conversion rate (proportion) tests
                effect_size = proportion_effectsize(
                    baseline_rate, 
                    baseline_rate * (1 + minimum_detectable_effect)
                )
                
                sample_size = tt_ind_solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1,
                    alternative='two-sided'
                )
                
            elif test_type == 'revenue':
                # For revenue (continuous) tests
                # Assume coefficient of variation of 1 for revenue
                cv = 1.0
                relative_effect = minimum_detectable_effect
                effect_size = relative_effect / cv
                
                sample_size = tt_ind_solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1,
                    alternative='two-sided'
                )
            
            elif test_type == 'retention':
                # For retention/churn tests
                effect_size = proportion_effectsize(
                    baseline_rate,
                    baseline_rate * (1 - minimum_detectable_effect)  # Reduction in churn
                )
                
                sample_size = tt_ind_solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1,
                    alternative='two-sided'
                )
            
            # Calculate test duration based on expected traffic
            result = {
                'sample_size_per_variant': int(np.ceil(sample_size)),
                'total_sample_size': int(np.ceil(sample_size * 2)),
                'effect_size': effect_size,
                'baseline_rate': baseline_rate,
                'expected_rate': baseline_rate * (1 + minimum_detectable_effect),
                'mde': minimum_detectable_effect,
                'alpha': alpha,
                'power': power
            }
            
            logger.info(f"Sample size calculated: {result['sample_size_per_variant']} per variant")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating sample size: {e}")
            raise CustomException("Failed to calculate sample size", e)
    
    def analyze_test_results(self, control_data, treatment_data, metric_type='conversion'):
        """Analyze A/B test results with multiple statistical tests"""
        try:
            logger.info(f"Analyzing A/B test results for {metric_type}...")
            
            results = {}
            
            if metric_type == 'conversion':
                # Binary outcome analysis
                control_conversions = control_data['converted'].sum()
                control_total = len(control_data)
                treatment_conversions = treatment_data['converted'].sum()
                treatment_total = len(treatment_data)
                
                # Conversion rates
                control_rate = control_conversions / control_total
                treatment_rate = treatment_conversions / treatment_total
                relative_lift = (treatment_rate - control_rate) / control_rate
                
                # Z-test for proportions
                z_stat, p_value = proportions_ztest(
                    [treatment_conversions, control_conversions],
                    [treatment_total, control_total]
                )
                
                # Confidence intervals
                se_control = np.sqrt(control_rate * (1 - control_rate) / control_total)
                se_treatment = np.sqrt(treatment_rate * (1 - treatment_rate) / treatment_total)
                
                ci_control = (control_rate - 1.96 * se_control, control_rate + 1.96 * se_control)
                ci_treatment = (treatment_rate - 1.96 * se_treatment, treatment_rate + 1.96 * se_treatment)
                
                # Bayesian analysis
                alpha_prior = 1
                beta_prior = 1
                
                alpha_control = alpha_prior + control_conversions
                beta_control = beta_prior + control_total - control_conversions
                alpha_treatment = alpha_prior + treatment_conversions
                beta_treatment = beta_prior + treatment_total - treatment_conversions
                
                # Probability that treatment is better
                samples = 10000
                control_samples = np.random.beta(alpha_control, beta_control, samples)
                treatment_samples = np.random.beta(alpha_treatment, beta_treatment, samples)
                prob_treatment_better = (treatment_samples > control_samples).mean()
                
                results = {
                    'control_rate': control_rate,
                    'treatment_rate': treatment_rate,
                    'relative_lift': relative_lift,
                    'absolute_lift': treatment_rate - control_rate,
                    'p_value': p_value,
                    'z_statistic': z_stat,
                    'control_ci': ci_control,
                    'treatment_ci': ci_treatment,
                    'prob_treatment_better': prob_treatment_better,
                    'is_significant': p_value < 0.05,
                    'sample_size_control': control_total,
                    'sample_size_treatment': treatment_total
                }
                
            elif metric_type == 'revenue':
                # Continuous outcome analysis
                control_revenue = control_data['revenue']
                treatment_revenue = treatment_data['revenue']
                
                # Remove outliers (optional)
                control_revenue_clean = control_revenue[
                    (control_revenue > control_revenue.quantile(0.01)) & 
                    (control_revenue < control_revenue.quantile(0.99))
                ]
                treatment_revenue_clean = treatment_revenue[
                    (treatment_revenue > treatment_revenue.quantile(0.01)) & 
                    (treatment_revenue < treatment_revenue.quantile(0.99))
                ]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(treatment_revenue_clean, control_revenue_clean)
                
                # Mann-Whitney U test (non-parametric alternative)
                u_stat, p_value_mw = stats.mannwhitneyu(
                    treatment_revenue, control_revenue, alternative='two-sided'
                )
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(control_revenue_clean) - 1) * control_revenue_clean.std() ** 2 +
                     (len(treatment_revenue_clean) - 1) * treatment_revenue_clean.std() ** 2) /
                    (len(control_revenue_clean) + len(treatment_revenue_clean) - 2)
                )
                cohens_d = (treatment_revenue_clean.mean() - control_revenue_clean.mean()) / pooled_std
                
                # Bootstrap confidence intervals
                bootstrap_diffs = []
                for _ in range(1000):
                    control_sample = np.random.choice(control_revenue_clean, size=len(control_revenue_clean), replace=True)
                    treatment_sample = np.random.choice(treatment_revenue_clean, size=len(treatment_revenue_clean), replace=True)
                    bootstrap_diffs.append(treatment_sample.mean() - control_sample.mean())
                
                ci_diff = np.percentile(bootstrap_diffs, [2.5, 97.5])
                
                results = {
                    'control_mean': control_revenue_clean.mean(),
                    'treatment_mean': treatment_revenue_clean.mean(),
                    'control_median': control_revenue.median(),
                    'treatment_median': treatment_revenue.median(),
                    'relative_lift': (treatment_revenue_clean.mean() - control_revenue_clean.mean()) / control_revenue_clean.mean(),
                    'absolute_lift': treatment_revenue_clean.mean() - control_revenue_clean.mean(),
                    'p_value_ttest': p_value,
                    'p_value_mannwhitney': p_value_mw,
                    't_statistic': t_stat,
                    'u_statistic': u_stat,
                    'cohens_d': cohens_d,
                    'ci_difference': ci_diff,
                    'is_significant': p_value < 0.05,
                    'sample_size_control': len(control_revenue_clean),
                    'sample_size_treatment': len(treatment_revenue_clean)
                }
            
            # Calculate statistical power
            if results['is_significant']:
                observed_effect_size = abs(results.get('cohens_d', results.get('relative_lift', 0)))
                observed_power = self.power_analysis.solve_power(
                    effect_size=observed_effect_size,
                    nobs1=results['sample_size_control'],
                    alpha=0.05
                )
                results['observed_power'] = observed_power
            
            logger.info(f"Test analysis completed. Significant: {results['is_significant']}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing test results: {e}")
            raise CustomException("Failed to analyze test results", e)
    
    def sequential_testing(self, control_data, treatment_data, metric_type='conversion',
                          alpha_spending_function='obrien_fleming'):
        """Implement sequential testing with alpha spending"""
        try:
            logger.info("Performing sequential testing analysis...")
            
            # Calculate information fraction
            planned_sample_size = 10000  # This should be pre-specified
            current_sample_size = len(control_data) + len(treatment_data)
            information_fraction = current_sample_size / planned_sample_size
            
            # O'Brien-Fleming spending function
            if alpha_spending_function == 'obrien_fleming':
                if information_fraction <= 0:
                    alpha_spent = 0
                else:
                    alpha_spent = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - 0.025) / np.sqrt(information_fraction)))
            else:
                # Pocock spending function
                alpha_spent = 0.05 * np.log(1 + (np.e - 1) * information_fraction)
            
            # Adjust significance threshold
            adjusted_alpha = alpha_spent
            
            # Perform standard analysis
            results = self.analyze_test_results(control_data, treatment_data, metric_type)
            
            # Update with sequential testing info
            results['information_fraction'] = information_fraction
            results['adjusted_alpha'] = adjusted_alpha
            results['is_significant_sequential'] = results['p_value'] < adjusted_alpha
            results['can_stop_early'] = results['is_significant_sequential']
            
            logger.info(f"Sequential testing: Information fraction {information_fraction:.2%}, "
                       f"Adjusted alpha {adjusted_alpha:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sequential testing: {e}")
            raise CustomException("Failed to perform sequential testing", e)
    
    def multi_metric_analysis(self, control_data, treatment_data, metrics):
        """Analyze multiple metrics with correction for multiple comparisons"""
        try:
            logger.info(f"Analyzing {len(metrics)} metrics...")
            
            results = {}
            p_values = []
            
            # Analyze each metric
            for metric_name, metric_config in metrics.items():
                metric_type = metric_config['type']
                metric_column = metric_config['column']
                
                if metric_type == 'binary':
                    # Create binary outcome data
                    control_metric = pd.DataFrame({'converted': control_data[metric_column]})
                    treatment_metric = pd.DataFrame({'converted': treatment_data[metric_column]})
                else:
                    # Continuous metric
                    control_metric = pd.DataFrame({'revenue': control_data[metric_column]})
                    treatment_metric = pd.DataFrame({'revenue': treatment_data[metric_column]})
                
                metric_results = self.analyze_test_results(
                    control_metric, treatment_metric, 
                    'conversion' if metric_type == 'binary' else 'revenue'
                )
                
                results[metric_name] = metric_results
                p_values.append(metric_results['p_value'])
            
            # Multiple comparison correction
            # Bonferroni correction
            bonferroni_alpha = 0.05 / len(metrics)
            
            # Benjamini-Hochberg FDR correction
            sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
            fdr_significant = []
            
            for i, (original_idx, p_val) in enumerate(sorted_p):
                threshold = 0.05 * (i + 1) / len(metrics)
                if p_val <= threshold:
                    fdr_significant.append(original_idx)
            
            # Update results with corrections
            for i, (metric_name, metric_results) in enumerate(results.items()):
                metric_results['bonferroni_significant'] = p_values[i] < bonferroni_alpha
                metric_results['fdr_significant'] = i in fdr_significant
                metric_results['bonferroni_alpha'] = bonferroni_alpha
            
            # Overall test decision
            overall_result = {
                'any_significant_uncorrected': any(r['is_significant'] for r in results.values()),
                'any_significant_bonferroni': any(r['bonferroni_significant'] for r in results.values()),
                'any_significant_fdr': len(fdr_significant) > 0,
                'num_metrics_tested': len(metrics),
                'metrics_results': results
            }
            
            logger.info(f"Multi-metric analysis completed. "
                       f"Significant metrics: {len(fdr_significant)}/{len(metrics)}")
            
            return overall_result
            
        except Exception as e:
            logger.error(f"Error in multi-metric analysis: {e}")
            raise CustomException("Failed to perform multi-metric analysis", e)
    
    def calculate_test_duration(self, required_sample_size, daily_traffic, 
                               test_allocation=0.5, include_weekends=True):
        """Calculate expected test duration"""
        try:
            logger.info("Calculating test duration...")
            
            # Adjust for test allocation
            daily_test_traffic = daily_traffic * test_allocation * 2  # Both variants
            
            # Calculate days needed
            days_needed = np.ceil(required_sample_size * 2 / daily_test_traffic)
            
            # Adjust for weekends if needed
            if not include_weekends:
                weeks = days_needed / 7
                weekend_days = weeks * 2
                days_needed = days_needed + weekend_days
            
            # Add buffer for variability
            buffer_multiplier = 1.2
            days_with_buffer = int(np.ceil(days_needed * buffer_multiplier))
            
            # Calculate end date
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_with_buffer)
            
            duration_info = {
                'days_needed': int(days_needed),
                'days_with_buffer': days_with_buffer,
                'weeks_needed': np.ceil(days_with_buffer / 7),
                'start_date': start_date.strftime('%Y-%m-%d'),
                'expected_end_date': end_date.strftime('%Y-%m-%d'),
                'daily_traffic': daily_traffic,
                'test_allocation': test_allocation
            }
            
            logger.info(f"Test duration: {days_with_buffer} days")
            return duration_info
            
        except Exception as e:
            logger.error(f"Error calculating test duration: {e}")
            raise CustomException("Failed to calculate test duration", e)
    
    def simulate_test_scenarios(self, baseline_rate, sample_size, num_simulations=1000):
        """Simulate A/B test scenarios to understand variability"""
        try:
            logger.info(f"Simulating {num_simulations} test scenarios...")
            
            scenarios = []
            
            for effect_size in [0, 0.05, 0.10, 0.20]:
                treatment_rate = baseline_rate * (1 + effect_size)
                significant_tests = 0
                observed_lifts = []
                
                for _ in range(num_simulations):
                    # Simulate data
                    control_conversions = np.random.binomial(sample_size, baseline_rate)
                    treatment_conversions = np.random.binomial(sample_size, treatment_rate)
                    
                    # Calculate rates
                    control_rate_sim = control_conversions / sample_size
                    treatment_rate_sim = treatment_conversions / sample_size
                    
                    # Test significance
                    _, p_value = proportions_ztest(
                        [treatment_conversions, control_conversions],
                        [sample_size, sample_size]
                    )
                    
                    if p_value < 0.05:
                        significant_tests += 1
                    
                    observed_lift = (treatment_rate_sim - control_rate_sim) / control_rate_sim
                    observed_lifts.append(observed_lift)
                
                scenarios.append({
                    'true_effect': effect_size,
                    'power': significant_tests / num_simulations,
                    'avg_observed_lift': np.mean(observed_lifts),
                    'lift_std': np.std(observed_lifts),
                    'lift_25th': np.percentile(observed_lifts, 25),
                    'lift_75th': np.percentile(observed_lifts, 75)
                })
            
            logger.info("Test simulation completed")
            return pd.DataFrame(scenarios)
            
        except Exception as e:
            logger.error(f"Error simulating test scenarios: {e}")
            raise CustomException("Failed to simulate test scenarios", e)
    
    def create_test_report(self, test_name, test_config, test_results, output_path="artifacts/ab_tests"):
        """Create comprehensive A/B test report"""
        try:
            logger.info(f"Creating test report for {test_name}...")
            
            import os
            os.makedirs(output_path, exist_ok=True)
            
            # Create visualizations
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Conversion Rates', 'Confidence Intervals', 
                              'Daily Performance', 'Cumulative Results')
            )
            
            # Add plots based on test results
            if 'control_rate' in test_results:
                # Binary metric visualization
                rates = [test_results['control_rate'], test_results['treatment_rate']]
                errors = [
                    test_results['control_ci'][1] - test_results['control_rate'],
                    test_results['treatment_ci'][1] - test_results['treatment_rate']
                ]
                
                fig.add_trace(
                    go.Bar(x=['Control', 'Treatment'], y=rates, error_y=dict(type='data', array=errors)),
                    row=1, col=1
                )
            
            # Save report
            report = {
                'test_name': test_name,
                'test_config': test_config,
                'test_results': test_results,
                'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            import json
            with open(f"{output_path}/{test_name}_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Create summary
            summary = f"""
A/B Test Report: {test_name}
Generated: {report['report_date']}

Test Configuration:
- Sample Size: {test_config.get('sample_size', 'N/A')} per variant
- Significance Level: {test_config.get('alpha', 0.05)}
- Power: {test_config.get('power', 0.8)}

Results:
- Control Performance: {test_results.get('control_rate', test_results.get('control_mean', 'N/A'))}
- Treatment Performance: {test_results.get('treatment_rate', test_results.get('treatment_mean', 'N/A'))}
- Relative Lift: {test_results.get('relative_lift', 0)*100:.1f}%
- P-value: {test_results.get('p_value', 'N/A')}
- Significant: {test_results.get('is_significant', False)}

Recommendation: {'Implement treatment' if test_results.get('is_significant', False) and test_results.get('relative_lift', 0) > 0 else 'Continue with control'}
"""
            
            with open(f"{output_path}/{test_name}_summary.txt", 'w') as f:
                f.write(summary)
            
            logger.info(f"Test report created: {output_path}/{test_name}_report.json")
            return report
            
        except Exception as e:
            logger.error(f"Error creating test report: {e}")
            raise CustomException("Failed to create test report", e)

class ABTestCalculator:
    """Streamlit-compatible A/B test calculator"""
    
    def __init__(self):
        self.framework = ABTestingFramework()
    
    def calculate_sample_size_simple(self, baseline_rate, mde_percent, confidence=95, power=80):
        """Simple interface for sample size calculation"""
        alpha = (100 - confidence) / 100
        power_decimal = power / 100
        mde_decimal = mde_percent / 100
        
        return self.framework.calculate_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=mde_decimal,
            alpha=alpha,
            power=power_decimal,
            test_type='conversion'
        )
    
    def analyze_results_simple(self, control_visitors, control_conversions,
                             treatment_visitors, treatment_conversions):
        """Simple interface for results analysis"""
        # Create mock data
        control_data = pd.DataFrame({
            'converted': [1] * control_conversions + [0] * (control_visitors - control_conversions)
        })
        treatment_data = pd.DataFrame({
            'converted': [1] * treatment_conversions + [0] * (treatment_visitors - treatment_conversions)
        })
        
        return self.framework.analyze_test_results(control_data, treatment_data, 'conversion')
    
    def get_duration_estimate(self, required_sample_size, daily_visitors):
        """Estimate test duration"""
        return self.framework.calculate_test_duration(
            required_sample_size=required_sample_size,
            daily_traffic=daily_visitors,
            test_allocation=0.5
        )

if __name__ == "__main__":
    # Example usage
    framework = ABTestingFramework()
    
    # Calculate sample size
    sample_size = framework.calculate_sample_size(
        baseline_rate=0.05,
        minimum_detectable_effect=0.20,
        alpha=0.05,
        power=0.8
    )
    print(f"Required sample size: {sample_size['sample_size_per_variant']} per variant")
    
    # Simulate test data
    np.random.seed(42)
    control_data = pd.DataFrame({
        'converted': np.random.binomial(1, 0.05, 5000),
        'revenue': np.random.gamma(50, 2, 5000)
    })
    treatment_data = pd.DataFrame({
        'converted': np.random.binomial(1, 0.06, 5000),
        'revenue': np.random.gamma(55, 2, 5000)
    })
    
    # Analyze results
    results = framework.analyze_test_results(control_data, treatment_data, 'conversion')
    print(f"Test results: Lift = {results['relative_lift']*100:.1f}%, p-value = {results['p_value']:.4f}")