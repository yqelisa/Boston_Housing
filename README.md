# Overview
This project **focuses** on demonstrating the algorithms of 3 types of models:

1. Linear Regression
2. Decision Tree
3. Neural Network

The **modeling problem** of this project is: given 13 geographical, social, and economics attributes, how do we predict the median housing price of a specific location in Boston.

The **value** of this project is, although there are many articles comparing model performances on the same dataset, not so many articles break down the implementation of the models by this level of detail. Whenever possible, I use base Python to implement functions for this project. Packages `numpy` and `pandas` are used for data manipulation.

# Data Sources
Boston Housing dataset is widely used in the Data Science community to benchmark model performance. 

Because it's a small dataset, it's easier to visualize and deep dive. While this dataset is small, the performance of Neural Network model on this dataset is not bad, making it ideal for demonstration.

I picked this dataset also because of my personal interest in Finance.

The data can be downloaded from: https://faculty.tuck.dartmouth.edu/images/uploads/faculty/business-analytics/Boston_Housing.xlsx

Attributes being used: 
- CRIM - per capita crime rate by town  
- ZN - proportion of residential land zoned for lots over 25,000 sq.ft.  
- INDUS - proportion of non-retail business acres per town.  
- CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
- NOX - nitric oxides concentration (parts per 10 million)  
- RM - average number of rooms per dwelling  
- AGE - proportion of owner-occupied units built prior to 1940  
- DIS - weighted distances to five Boston employment centres  
- RAD - index of accessibility to radial highways  
- TAX - full-value property-tax rate per $10,000  
- PTRATIO - pupil-teacher ratio by town    
- LSTAT - % lower status of the population  
- MEDV - Median value of owner-occupied homes in $1000's

# Set Up
The following **Python** packages are required to run the `.ipynb` notebook:
- `pandas` - dataframe manipulation
- `numpy` - data series manipulation and calculation
- `matplotlib` - visualization
- `copy` - create deep copies of objects
- `math` - calculation
- `sklearn` - models and preprocessing tools ready to use
- `tensorflow` - for Neural Networks

# Files
- *Boston_Housing.ipynb* - notebook that includes code for data manipulation and calculation, with visualization and modeling results
- *Boston_Housing.xlsx* - source data

# Results
Although the initial attempts of applying the 3 models yield different costs, through optimization, the models achieved very similar results.

|Model|Cost (MSE)|
|-------|-------:|
|Linear Regression|34.3951|
|Decision Tree|30.8287|
|Neural Network|30.5800|

# References
- Xavier Bourret Sicotte. (2018). [Lasso regression: implementation of coordinate descent](https://xavierbourretsicotte.github.io/lasso_implementation.html)
- D@KG. (2022). [Coordinate Descent for LASSO & Normal Regression](https://www.kaggle.com/code/ddatad/coordinate-descent-for-lasso-normal-regression)
- Lari Giba. [Lasso Regression Explained, Step by Step](https://machinelearningcompass.com/machine_learning_models/lasso_regression/)
- Lari Giba. [Ridge Regression Explained, Step by Step](https://machinelearningcompass.com/machine_learning_models/ridge_regression/)
- Berat Yildirim. (2022). [Regression Tree From Scratch Using Python](https://medium.com/@beratyildirim/regression-tree-from-scratch-using-python-a74dba2bba5f)
- Harkirat Vasir. (2020). [Boston Housing|Neural Network|Beginners tutorial](https://www.kaggle.com/code/harkiratvasir/boston-housing-neural-network-beginners-tutorial)
- Arunkumar Venkataramanan. (2019). [TensorFlow Tutorial and Housing Price Prediction](https://www.kaggle.com/code/arunkumarramanan/tensorflow-tutorial-and-housing-price-prediction)
- Coursera Deep Learning course, sourced from Kamyar Nazeri. (2018). [dnn_utils_v2.py](https://github.com/knazeri/coursera/blob/master/deep-learning/1-neural-networks-and-deep-learning/4-building-your-deep-neural-network-step-by-step/dnn_utils_v2.py)

# Acknowledgements
