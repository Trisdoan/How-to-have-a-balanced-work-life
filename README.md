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

### Check distribution

Most of features are not normal distrbuted. There are 2 binary distribution: sufficient income and BMI_range

```python

df.hist(figsize=(20,20))

 ```
 
 
 ### Categorical exploration


Seems like most people are from 21 to 50 anticipated in the survey. More female than male

```python
# Gender and Age
pd.crosstab(df.GENDER,df.AGE, normalize=True).plot(kind="bar")
 ```



Again, rate of female is higher than male. It's also because of higher number of female anticipating

```python
# BMI and gender
pd.crosstab(df.GENDER,df.BMI_RANGE, normalize=True).plot(kind="bar")
 ```
 
 

```python
# Sufficient and gender
## Sufficient means: HOW SUFFICIENT IS YOUR INCOME TO COVER BASIC LIFE EXPENSES
pd.crosstab(df.GENDER,df.SUFFICIENT_INCOME, normalize=True).plot(kind="bar")
 ```

## PREDICTION MODEL

### Feature Transformation & Selection

#### Replace values for easy understanding

```python
df.BMI_RANGE.replace({1: "BMI < 25", 2: "BMI > 25"}, inplace=True)
df.SUFFICIENT_INCOME.replace({1: "Not or hardly sufficient", 2: "Sufficient"}, inplace=True)
 ```

#### Get dummies categorical features

```python
BMI = pd.get_dummies(df.BMI_RANGE,drop_first=True)
INCOME = pd.get_dummies(df.SUFFICIENT_INCOME,drop_first=True)
AGE = pd.get_dummies(df.AGE,drop_first=True)
GENDER = pd.get_dummies(df.GENDER,drop_first=True)
 ```
 
 
#### DROP unessarry columns

```python
df.drop(["Timestamp","GENDER","BMI_RANGE","SUFFICIENT_INCOME","AGE"],axis=1,inplace=True)
 ```

#### CONCATENATE created features

```python
df = pd.concat([df, BMI, INCOME, AGE, GENDER],axis=1)
 ```

#### Skewness transformation

##### Check skewness 

```python
skewness_dict = {}
for column in df.columns:
    if df[column].dtypes == "int64" or df[column].dtypes == "float64":
        skewness_dict[column] = df[column].skew()
    else:
        continue
## Sort based on absolute value because which values above 0.25 are considered highly skewed
skewness = sorted(skewness_dict.items(),key=lambda element: abs(element[1]), reverse=True)
sns.set(rc={'figure.figsize': (15,10)})

#assign x and y for barplot
x_1 = []
for row in skewness:
    value = row[0]
    x_1.append(value)

y_1 = []
for row in skewness:
    value = row[1]
    y_1.append(value)

#plot 
plot = sns.barplot(x=x_1, y=y_1)
for item in plot.get_xticklabels():
    item.set_rotation(90)

plot.set_title("Feature Skewness ", fontsize = 16)
plot.set_xlabel("Features", fontsize = 12)
plot.set_ylabel("Skewness", fontsize = 12)
 ```

##### Transform skewness data using yeo-johnson

```python
from scipy import stats
from scipy.stats import yeojohnson
transformed_skew = {}
parameters_skew = {}
for col in x_1[:8]:
    transformed_skew[col + "_transformed"], parameters = stats.yeojohnson(df[col])
    parameters_skew[col+"_transformed"] = parameters

transformed_df = pd.DataFrame(transformed_skew)
transformed_df.head()
 ```

##### Re-check skewness

```python
skewness_dict_transformed = {}
for column in transformed_df.columns:
        skewness_dict_transformed[column] = transformed_df[column].skew()
## Sort based on absolute value because which values above 0.25 are considered highly skewed
skewness_transformed = sorted(skewness_dict_transformed.items(),key=lambda element: abs(element[1]), reverse=True)
sns.set(rc={'figure.figsize': (15,10)})

#assign x and y for barplot
x_2 = []
for row in skewness_transformed:
    value = row[0]
    x_2.append(value)

y_2 = []
for row in skewness_transformed:
    value = row[1]
    y_2.append(value)

#plot 
plot = sns.barplot(x=x_2, y=y_2)
for item in plot.get_xticklabels():
    item.set_rotation(90)

plot.set_title("Feature Post Skewness ", fontsize = 16)
plot.set_xlabel("Features", fontsize = 12)
plot.set_ylabel("Skewness", fontsize = 12)
 ```

#### Split into 2 datasets

```python
# skewed_df
skewed_df = pd.concat((df.drop(columns = [col for col in x_1[:8]]),transformed_df), axis=1)
skewed_df

#non_skew_df
df
 ```


### Check Multicollinearity by VIF

```python
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
# def calc VIF
def cal_vif(X):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    return vif
# vif for non_skew df
vif_df = cal_vif(df.drop(columns = "WORK_LIFE_BALANCE_SCORE"))
vif_df.sort_values(by="VIF Factor", ascending=False)
 ```


```python
# vif for non_skew df
vif_skewed_df = cal_vif(skewed_df.drop(columns = "WORK_LIFE_BALANCE_SCORE"))
vif_skewed_df.sort_values(by="VIF Factor", ascending=False)
 ```


#### Re-calc VIF

```python
# vif for non_skew df which removed columns having VIF above 5.5
vif_df_2 = cal_vif(df.drop(columns = ["WORK_LIFE_BALANCE_SCORE", "SLEEP_HOURS", "TODO_COMPLETED", 
                                    "SOCIAL_NETWORK", "FRUITS_VEGGIES","WEEKLY_MEDITATION",
                                    "SUPPORTING_OTHERS", "PERSONAL_AWARDS"], axis=1))
vif_df_2.sort_values(by="VIF Factor", ascending=False)
 ```

```python
# vif for non_skew df which removed columns having VIF above 5.5
vif_skewed_df_2 = cal_vif(skewed_df.drop(columns = ["WORK_LIFE_BALANCE_SCORE", "SLEEP_HOURS_transformed", "ACHIEVEMENT_transformed", 
                                    "SOCIAL_NETWORK", "FLOW_transformed","TODO_COMPLETED_transformed",
                                    "FRUITS_VEGGIES", "TIME_FOR_PASSION_transformed", "WEEKLY_MEDITATION",
                                    "SUPPORTING_OTHERS"], axis=1))
vif_skewed_df_2.sort_values(by="VIF Factor", ascending=False)
 ```


## Train model

### Create X and Y for 2 datasets

```python
# preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

x_1 = df.drop(columns = ["WORK_LIFE_BALANCE_SCORE", "SLEEP_HOURS", "TODO_COMPLETED", 
                                    "SOCIAL_NETWORK", "FRUITS_VEGGIES","WEEKLY_MEDITATION",
                                    "SUPPORTING_OTHERS", "PERSONAL_AWARDS"], axis=1)
y_1 = df.WORK_LIFE_BALANCE_SCORE

x_2 = skewed_df.drop(columns = ["WORK_LIFE_BALANCE_SCORE", "SLEEP_HOURS_transformed", "ACHIEVEMENT_transformed", 
                                    "SOCIAL_NETWORK", "FLOW_transformed","TODO_COMPLETED_transformed",
                                    "FRUITS_VEGGIES", "TIME_FOR_PASSION_transformed", "WEEKLY_MEDITATION",
                                    "SUPPORTING_OTHERS"], axis=1)
y_2 = skewed_df.WORK_LIFE_BALANCE_SCORE
 ```
 



### Scaling numeric features

```python
# our scaler
scaler = MinMaxScaler()
#scaler = StandardScaler()

# fit the scaler to our data
numeric_x_1 = x_1.drop(columns = ['BMI > 25', 'Sufficient',
       '36 to 50', '51 or more', 'Less than 20', 'Male'],axis =1 )

scaled_numeric_x_1 = pd.DataFrame(scaler.fit_transform(numeric_x_1), columns = numeric_x_1.columns)

x_1 = pd.concat((scaled_numeric_x_1,x_1[['BMI > 25', 'Sufficient',
       '36 to 50', '51 or more', 'Less than 20', 'Male']]),axis=1)
# describe the scaled data
x_1.describe()
 ```
 
### Check R2

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_1, y_1,random_state = 0,test_size=0.25)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_train)

print("R squared: {}".format(r2_score(y_true=y_train,y_pred=y_pred)))
 ```
 
### Chech Multivariate Normality

```python
residuals = y_train.values - y_pred

p = sns.distplot(residuals,kde=True)
p = plt.title('Normality of residuals')
 ```
 => Residual distribution is normall distributed. It's good
 
 
### Check Homoscedasticity


```python
p = sns.scatterplot(y_pred,residuals)
plt.xlabel('predicted values')
plt.ylabel('Residuals')
p = sns.lineplot([y_pred.min(),y_pred.max()],[0,0],color='blue')
 ```
=> There is no clear pattern between residuals and predicted values. It's good


### Check Homoscedasticity by Goldeld Quantdt test


```python
# H0: Error terms are homoscedastic
# H1: The Error terms are heteroscedastic

import statsmodels.stats.api as sms
from statsmodels.compat import lzip
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(residuals, X_train)
lzip(name, test)
 ```
 ('F statistic', 0.9651359360560795), ('p-value', 0.9148329857291659)

=> There is no sufficient evidence to reject the null. It's good



## Applying model to test dataset

```python
pred_y = regr.predict(X_test)

R2_test = regr.score(X_test,y_test) 
from matplotlib import pyplot as plt
plt.scatter(y_test, pred_y, alpha = 0.2)
plt.xlabel('Work Life Balance (actual)', size = 16)
plt.ylabel('Predicted values', size = 16)

plt.title('Model Trained R Squared ='+ '{number:.3f}'.format(number=R2_test), size = 20)
 ```
=> Look good!!!

### Check residuals
```python
residual_df = pd.DataFrame(pred_y, columns = ['Predicted'])
y_test = y_test.reset_index (drop = True)
residual_df["Target"] = y_test
residual_df["Residual"] = residual_df["Target"] - residual_df["Predicted"]
residual_df["Residual%"] = abs((residual_df["Target"] - residual_df["Predicted"])/residual_df["Target"]*100)
residual_df.describe()
 ```
=> In worste case, max residual percentage is 7%. It means expecting standard deviation to be 9% different from actual values


## Check feature weight: which features drive target variable the most ?

```python
reg_summary = pd.DataFrame(x_1.columns.values, columns = ["Features"])
reg_summary["Weights"] = regr.coef_

# plot bar chart
f, ax = plt.subplots(figsize=(15, 6))
sns.barplot(x="Weights", y="Features", data=reg_summary.sort_values("Weights", ascending=False, key = abs),
            label="Weights")
ax.set_title("Feature Weights in Linear Regression",fontsize=20)      
 ```















 
