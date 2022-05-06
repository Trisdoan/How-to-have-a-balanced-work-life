# GOAL OF PROJECT

I want to know which factors affecting our life, whether or not based on some factors to predict quality of life

Link dataset: https://www.kaggle.com/datasets/ydalat/lifestyle-and-wellbeing-data/code

I refered to this github: https://github.com/taufiqbashori/wellbeing-regression/blob/main/Work_Life_Balance_MultiRegression%20(1).ipynb


# INSIGHTS 

# ANALYTICS PROCESS

## IMPORT LIBRARIES & DATASET

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option("display.max_column",None)

data = pd.read_csv("Wellbeing_and_lifestyle_data_Kaggle.csv")
df = data.copy()
 ```

## CLEANING & EDA

### Transfer non-numeric datatype into numeric

I used to_numeric of pandas to transform. There are nulls so I used parameter: errors="coerce"

```python
# using to_numeric with errors = coerce to transfer non-numeric into nan
df.DAILY_STRESS = pd.to_numeric(df.DAILY_STRESS,errors="coerce")
 ```
 
### Fill daily_stress with median

There are not too much nills in this column. I used common approach "median" to replace those nulls

```python
#Check null or not
df.DAILY_STRESS.isnull().sum()

#fill with median
df.DAILY_STRESS.fillna(df.DAILY_STRESS.median(), inplace=True)
 ```


## PREDICTION MODEL

