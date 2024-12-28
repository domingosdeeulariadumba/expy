# Expy Python Module

This module aims to make it easier for data scientists, analysts, and engineers to conduct statistically sound experiments. In the core of this module is the `ABTesting` class. This class provides tools for designing, analyzing, and simulating A/B tests.

---

## Features

- **Sample Size Calculation**: Calculate the minimum required sample size using Evan Miller's methodology.
- **Simulation of Experiment Results**: Simulate control and treatment group results for various scenarios.
- **Result Analysis**: Retrieve and visualize experiment results with support for confidence intervals and kernel density estimates (KDEs).

---

## Dependencies

The following Python libraries are required:

```python
import scipy.stats as scs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

## Class Overview

### `ABTesting`
The main class provides the following methods and attributes:

#### **Initialization**
```python
ABTesting(bcr, mde, alpha=0.05, power=0.8, absolute_variation=True, two_tailed=True)
```

- **bcr**: Baseline Conversion Rate (0 < bcr < 1).
- **mde**: Minimum Detectable Effect (absolute or relative) (0 < mde < 1).
- **alpha**: Significance level (default: 0.05).
- **power**: Statistical power (default: 0.8).
- **absolute_variation**: Whether `mde` is absolute (default: True).
- **two_tailed**: Use two-tailed tests (default: True).

#### **Methods**

1. **`evan_miller_sample_size()`**
   - Calculates the required sample size using Evan Miller's methodology.

2. **`simulate_experiment_results(p_ctrl, n_ctrl_inc = 0, n_trmt_inc = 0, lift = 0.0, summary_table = True, random_state = None)`**
   - Simulates experiment results for control and treatment groups.
   - Parameters:
     - `p_ctrl`: Conversion rate in the control group.
     - `n_ctrl_inc`: Additional samples for the control group.
     - `n_trmt_inc`: Additional samples for the treatment group.
     - `lift`: Incremental difference between groups.
     - `summary_table`: Return aggregated summary table (default: True).
     - `random_state`: Set for reproducibility.

3. **`get_experiment_results(n_ctrl, p_ctrl, n_trmt, p_trmt, plot_type = 'KDE')`**
   - Analyzes and visualizes results.
   - Parameters:
     - `n_ctrl`, `n_trmt`: Sample sizes of the control and treatment groups.
     - `p_ctrl`, `p_trmt`: Conversion rates for control and treatment groups.
     - `plot_type`: Visualization method (`'KDE'` or `'Confidence Intervals'`).

---

## Installation and importing the module

````python
pip install expy
````

```python
from expy import ABTesting
```

---

## Usage

### Example: Calculate Sample Size
```python
ab_test = ABTesting(bcr = 0.1, mde = 0.02, alpha = 0.05, power = 0.8)
sample_size = ab_test.evan_miller_sample_size()
print(f"Required Sample Size: {sample_size}")
```

### Example: Simulate Experiment Results
```python
simulated_results = ab_test.simulate_experiment_results(p_ctrl = 0.1, lift = 0.02)
print(simulated_results)
```

### Example: Visualize Experiment Results
```python
ab_test.get_experiment_results(n_ctrl = 500, p_ctrl = 0.1, n_trmt = 500, p_trmt = 0.12, plot_type = 'Confidence Intervals')
```

💡 A more detailed example regarding the implementation of this tool is available <em> <a href = 'https://github.com/domingosdeeulariadumba/expy/blob/main/ExpyExamplesNotebook.ipynb' target = '_blank'> here.</em> 

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contribution

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## References
- FÁVERO, L. P.; BELFIORE, P. <em> <a href = 'https://www.amazon.com.br/Manual-An%C3%A1lise-Dados-Luiz-F%C3%A1vero/dp/8535270876' target = '_blank'> Manual de análise de dados: estatística e modelagem
multivariada com Excel®, SPSS® e Stata®.</em> Rio de Janeiro: Elsevier, 2017.
- GRAVETTER, F. J.; WALLNAU, L. B. <em> <a href = 'https://www.amazon.com/Statistics-Behavioral-Sciences-Standalone-Book/dp/1305504917' target = '_blank'> Statistics for the Behavioral Sciences.</em> 10th ed. Boston:
Cengage Learning, 2015.
- SAINANI K. Stanford University. <em> <a href = 'https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://web.stanford.edu/~kcobb/hrp259/lecture11.ppt&ved=2ahUKEwin0_6qmsuKAxVHUEEAHSzNEt0QFnoECBUQAQ&usg=AOvVaw16arOYUy8mK6FcYHGblX0m' target = '_blank'> Introduction to Sample Size and Power Calculations</em>. Last accessed on Dec 28 2024.
- UDACITY. <em> <a href = 'https://www.udacity.com/course/ab-testing--ud257' target = '_blank'> A/B Testing</em>. Last accessed on Dec 28 2024.
  
___
## Acknowledgments

This project was mainly possible due to the contribution of Evan Miller regarding A/B testing methodologies and tools. Refer to his <em> <a href = 'https://www.evanmiller.org/ab-testing/sample-size.html' target = '_blank' a> A/B Testing Sample Size Calculator.</em> for more details.
