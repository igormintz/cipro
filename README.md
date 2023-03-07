# prediction of antimicrobial resistance to ciprofloxacin based on patients' electronic medical recored
This code was written as part of a research on antimicrobial resistance. [link to preprint](https://www.medrxiv.org/content/10.1101/2022.10.18.22281205v1)(will be replaced once published).

# Data

Data are proprietary but can be made available upon reasonable request from the authors.

The input data are csv files paths with numerical features. I used 2 version: bactria gnostic and bacteria agnostic.
```
csv_paths= {
            'Agnostic': get_script_dir() / 'agnostics_22_06_22.csv',
            'Gnostic': get_script_dir() / 'gnostics_22_06_22.csv',
            }
```
# The Algorithm
The model an ensemble of LASSO penalized logistic regression, random forest, gradient-boosted trees , and a simple neural network, stacked as described in [Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. Statistical applications in genetics and molecular biology, 6(1).](https://www.degruyter.com/document/doi/10.2202/1544-6115.1309/html)
The code contains a workaround of an issue with other ensemble packages that can't deal with time series split (while cross validating to create the z_matrix the number of predictions is smaller than the whole data set because the first fold is used for training only). More over, the code plots AUC-ROC, calibration plot, SHAPley values, and net benefit plots, and saves a csv file with results.

# Python Dependencies
described in a requirements.txt file

# Usage
After updating the csv path to the data, just run:
`python main.py` for the creation of the search space, creation of z-table and training. after it's done, run `python inference.py`. In the end, run `python calc_shap.py'

# Licens
MIT License
