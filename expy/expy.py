# Dependencies
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# A class for A/B tests
class ABTesting:
    '''
    This class provides methods and attributes for design, analysis, and
    simulation of A/B experiments.
    '''
    
    # Initializing the class
    def __init__(self, bcr: float, mde: float, alpha: float = .05, power: float = .8, absolute_variation: bool = True, two_tailed: bool = True):
        '''
        Parameters
        ----------
        - bcr: the Baseline Conversion Rate.
        - mde: the Minimum Detectable Effect (or practical significance).
        - alpha: the Significance level of the experiment (default: 0.05).
        - power: statistical power — measures the probability that the test will
          reject the null hypothesis if the treatment really has an effect 
          (default: 0.8).
        - absolute_variation: whether the diffrence between the two groups is
          absolute or relative (default: True)
        - two_tailed: for deciding between a two or a one-tailed test (default: True)
        '''
        
        # Parameters range entries
        if not (0 < bcr < 1):
            raise ValueError('Baseline Conversion Rate (bcr) must be between 0 and 1.')
        if not (0 < mde < 1):
            raise ValueError(' Minimum Detectable Effect (mde) must be between 0 and 1.')
        if not (0 < alpha < 1):
            raise ValueError('Significance level (alpha) must be between 0 and 1.')
        if not (0 < power < 1):
            raise ValueError('Power must be between 0 and 1.')
                
        # Attributes
        self.bcr = bcr
        self.effect_size = mde if absolute_variation else bcr * mde
        self.alpha = alpha/2 if two_tailed else alpha
        self.power = power
       
        
    # Evan Miller Sample Size Calculator   
    def evan_miller_sample_size(self) -> int:        
        '''
        A method for retrieving the required sample size using Evan Miller's 
        methodology.
        '''       
            
        # Setting the variation type    
        p2 = self.bcr + self.effect_size
        q1, q2 = 1 - self.bcr, 1 - p2 
    
        # Z-scores for significance level and power
        z_alpha = scs.norm.ppf(1 - self.alpha)    
        z_power = scs.norm.ppf(self.power)
        
        # Calculating the standard deviations
        std1 = np.sqrt(2 * self.bcr * q1)
        std2 = np.sqrt(self.bcr * q1 + p2 * q2)
        
        # Computing the sample size per group
        sample_size = pow(
            (z_alpha * std1 + z_power * std2) / self.effect_size, 2
        )    
        return round(sample_size)

    
    # Experiment simulator
    def simulate_experiment_results(self, p_ctrl: float, n_ctrl_inc: int = 0, n_trmt_inc: int = 0, lift: float = .0, summary_table = True, random_state: int = None) -> pd.DataFrame:         
        """
        This method returns a dataframe simulating the experiment results given
        a minimum sample size.
        
        Parameters
        ----------
        - p_ctrl: the proportion of conversion in the control group.
        - n_ctrl_inc: incremental samples from the Control Group (the yardstick
          sample size  is the one obtained through the Evan Miller's methodology).
        - n_trmt_inc: additional samples from the treatment group.
        - lift: the incremental difference between the two groups (default = 0).
          In case it is equal to negative effect size there is no difference 
          between the control and experiment group.
        - summary_table (default: True): for deciding whether to return the raw
          or the aggregated DataFrame summarizing the experiment results.
        - random_state (default: None): for reproducibility of the simulation.
        """
        
        # Conditions for p_ctrl and lift
        if not (0 < p_ctrl < 1):
            raise ValueError('the proportion of conversion in the control group (p_ctrl) must be between 0 and 1.')
        if not (-1 < lift < 1):
            raise ValueError('Lift must be between -1 and 1.')
                
        # Condition for random state
        if random_state is not None:
            np.random.seed(random_state)
    
        # Computing the sample size
        baseline_sample_size = self.evan_miller_sample_size()
        n_ctrl = baseline_sample_size + n_ctrl_inc
        n_trmt = baseline_sample_size + n_trmt_inc
        p_trmt = p_ctrl + self.effect_size + lift
        
        # Calculating the experiment results and storing them in a DataFrame
        results_ctrl = np.random.binomial(n = 1, size = n_ctrl, p = p_ctrl)
        results_trmt = np.random.binomial(n = 1, size = n_trmt, p = p_trmt)
        exp_results_df = pd.concat([
            pd.DataFrame({'group': 'control', 'result': results_ctrl}),
            pd.DataFrame({'group': 'treatment', 'result': results_trmt})
                                       ]).sample(frac = 1)
        # Outcome conditions
        if not  summary_table:
            return exp_results_df 
        else:
            conversion_df = exp_results_df.groupby('group').mean()
            conversion_df['sample size'] = [n_ctrl, n_trmt]            
            return conversion_df.T

    
    # a method for experiment results summary
    def get_experiment_results(self, n_ctrl: int, p_ctrl: float, n_trmt: int, p_trmt: float, plot_type: str = 'KDE'):
        """
        Method for retrieving the experiment results.
        
        Parameters
        ----------
        - n_ctrl: the sample size of the control group.
        - n_trmt: the size of the treatment group.
        - p_ctrl: the proportion of conversion in the control group.
        - p_trmt: the proportion of conversion in the treatment group.
        - plot_type (default: 'KDE'): parameter for deciding whether to plot 
          KDEs or Confidence Intervals for supporting the final decision.
        """
        
        # Conditions for p_ctrl and lift
        if not (0 < p_ctrl < 1):
            raise ValueError('The proportion of conversion in the control group (p_ctrl) must be between 0 and 1.')
        if not (0 < p_trmt < 1):
            raise ValueError('The proportion of conversion in the tratment group (p_trmt) must be between 0 and 1.')
        
        # Computing the pooled Standard Error
        pooled_p = (n_ctrl * p_ctrl + n_trmt * p_trmt) / (n_ctrl + n_trmt)
        pooled_q = 1 - pooled_p
        pooled_se = np.sqrt(pooled_p * pooled_q * (1 / n_ctrl + 1 / n_trmt))        
        
        # Estimated difference between the two groups and its margin of error
        d_hat = p_trmt - p_ctrl
        norm_trmt = scs.norm(d_hat, pooled_se)
        d_hat_min, d_hat_max = norm_trmt.ppf(self.alpha), norm_trmt.ppf(1 - self.alpha)
       
        # Setting the confidence intervals for retaining the null hypothesis
        norm_ctrl = scs.norm(0, pooled_se)
        lower_bound, upper_bound = norm_ctrl.ppf(self.alpha), norm_ctrl.ppf(1 - self.alpha)
        
        # Plotting options
        if plot_type == 'KDE':
            x_trmt = np.linspace(d_hat - pooled_se * 5, d_hat + pooled_se * 5, int(n_trmt))
            y_trmt = norm_trmt.pdf(x_trmt)
            x_ctrl = np.linspace(- pooled_se * 5, pooled_se * 5, int(n_ctrl))
            y_ctrl = norm_ctrl.pdf(x_ctrl)
            
            plt.plot(x_ctrl, y_ctrl, 'r:')
            plt.fill_between(x_ctrl, 0, y_ctrl, color = 'r', alpha = .25, hatch = '//', 
                             where = (x_ctrl >= lower_bound) & (x_ctrl <= upper_bound), label = 'Control')
            plt.plot(x_trmt, y_trmt, 'c:')
            plt.fill_between(x_trmt, 0, y_trmt, color = 'c', alpha = .25, hatch = '//', 
                             where = (x_trmt >= d_hat_min) & (x_trmt <= d_hat_max), label = 'Treatment')
            plt.axvline(x = .05, linestyle = ':', color = '#04ef62', label = 'MDE')
            plt.yticks([])
            plt.ylim(0)
            plt.legend()
            plt.title('Experiment Results')
            plt.show()
        elif plot_type == 'Confidence Intervals':
            effect_size = self.effect_size
            plt.errorbar(0, 5, xerr = abs(lower_bound), fmt = 'o', 
                         linewidth = 2, capsize = 5, color = 'r', alpha = .5, label = 'Control')
            plt.errorbar(d_hat, 7, xerr = abs(d_hat - d_hat_min), 
                         fmt = 'o', linewidth = 2, capsize = 5, color = 'c', alpha = .5, label = 'Treatment')
            plt.plot([- effect_size, - effect_size], [3, 9], color = '#04ef62', linestyle = ':')
            plt.plot([effect_size, effect_size], [3, 9], color = '#04ef62', linestyle = ':')
            plt.text(effect_size * .9, 2.5, 'MDE', color = '#04ef62')
            plt.text(- effect_size * 1.2, 2.5, '- MDE', color = '#04ef62')
            plt.legend()
            plt.ylim(2, 10)
            plt.xlim(-.05 * 1.25)
            plt.yticks([])
            plt.title('Experiment Results')
            plt.show()
        else:
            raise ValueError("There are only two alternatives for this parameter: 'KDE' and 'Confidence Intervals'!")
            
        # Setting recommendations based on experiment results
        if d_hat_min >= self.effect_size:
            recommendation = f'Given that {d_hat_min:.2f} (the lower bound for the estimated difference) ≥ {self.effect_size} (the practical significance), it is recommended to launch the version B!'
        elif d_hat_max <= self.effect_size:
            recommendation = f'Since {d_hat_max: .2f} (the upper bound for the estimated difference) ≤ {self.effect_size} (the practical significance), it is then recommended to keep the current version!'
        else:
            recommendation = 'There might not have enough Power to draw any conclusion about the experiment results. Thus, it is recommended to conduct some additional tests.'
        print(f'\n\n=======================================================================\nNotes:\n[1] {recommendation}')
