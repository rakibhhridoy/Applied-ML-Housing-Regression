# *Applied Machine Learning Housing- Regression*

![img0](https://miro.medium.com/max/566/1*Zm2Hu724W6UQCVWWQe7afg.jpeg)


>1.Understand the requirements of the business

We are enthusiastic data scientists and before starting we need to ask some fundamental questions

 1. Why does our organisation need this predictive model?

    * possibly we are a real-estate firm and interested in investing in California
    * the organisation will use this data to feed another machine learning model
    * current process is good but manual and time consuming
    * our organisation wants an edge over competition
    * we are a consulting firm in the real-estate business and this data is valuable

2. We need to understand what are we doing at the root level

    * We’ll train our model on existing data so we are doing supervised learning
    * Since we need to predict housing prices we are doing regression
    * Output depends on many parameters so we are doing multivariate-regression

>2. Acquire the dataset
Get the dataset in CSV format here and store it in a folder. We prepare a virtual environment, activate it, install the dependencies
Start Jupyter notebook and do the basic imports
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pdhousing = pd.read_csv('./housing.csv')
housing.head(5)
```

This data has metrics such as the ```population, median income, median housing price```, and so on for each block group in California.

```python
housing.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
longitude             20640 non-null float64
latitude              20640 non-null float64
housing_median_age    20640 non-null float64
total_rooms           20640 non-null float64
total_bedrooms        20433 non-null float64
population            20640 non-null float64
households            20640 non-null float64
median_income         20640 non-null float64
median_house_value    20640 non-null float64
ocean_proximity       20640 non-null object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
```
next you will find in notebook. 
[Source](https://medium.com/@gurupratap.matharu/end-to-end-machine-learning-project-on-predicting-housing-prices-using-regression-7ab7832840ab)

# Get Touch With Me
Connect- [Linkedin](https://linkedin.com/in/rakibhhridoy)
Website- [RakibHHridoy](https://rakibhhridoy.github.io)


 
