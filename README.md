# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from category_encoders import BinaryEncoder
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer
data=pd.read_csv(r"C:\Users\acer\Downloads\data.csv")
df=pd.DataFrame(data)
df
```
<img width="570" height="368" alt="image" src="https://github.com/user-attachments/assets/4949304c-79d1-4ea1-8f3a-1c2696be022b" />


```
df.isnull()
```
<img width="485" height="358" alt="image" src="https://github.com/user-attachments/assets/4589f2c2-3ba3-4ea3-b48a-4a3acdb6d370" />


```
le=LabelEncoder()
df3=pd.DataFrame()
df3["City"]=df["City"]
df3["City_encoded"]=le.fit_transform(df['City'])
print(df3)
```

<img width="372" height="281" alt="image" src="https://github.com/user-attachments/assets/e719c2d9-c2d3-460e-8e42-afa84e16d707" />

```
df2=pd.DataFrame()
df2['Ord_2']=df["Ord_2"]
education=['High School','Diploma','Bachelors','Masters','PhD']
enc=OrdinalEncoder(categories=[education])
encoded=enc.fit_transform(df2[['Ord_2']])
df2['Encoded']=encoded
print(df2)
```

<img width="438" height="285" alt="image" src="https://github.com/user-attachments/assets/1b5eb4d5-7218-4285-b9ac-39a9ba14405d" />

```
df4=pd.DataFrame()
df4['Ord_1']=df["Ord_1"]
ohe=OneHotEncoder(sparse_output=False)
enc=pd.DataFrame(ohe.fit_transform(df[["Ord_1"]]))
df4=pd.concat([df4,enc],axis=1)
df4
```

<img width="292" height="366" alt="image" src="https://github.com/user-attachments/assets/7acf9e65-2f2c-401e-9459-28f1b931d2ac" />

```
df5=pd.DataFrame()
df5['City']=df['City']
be=BinaryEncoder(cols=['City'])
encoded=be.fit_transform(df['City'])
df5=pd.concat([df5,encoded],axis=1)
print(df5)
```

<img width="404" height="276" alt="image" src="https://github.com/user-attachments/assets/7fd221d0-326b-4e52-826c-439627fbcafa" />

```
df=pd.DataFrame(pd.read_csv(r"C:\Users\acer\Downloads\Data_to_Transform.csv"))
df
```

<img width="905" height="439" alt="image" src="https://github.com/user-attachments/assets/d0c403f8-6c2e-44e4-90b9-9fca5429f5d2" />

```
df['Highly PositiveSkew']=1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
```

<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/2a31f3f2-a949-4cc5-911c-fabba402a0ae" />

```
df['Moderate PositiveSkew']=np.sqrt(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

```

<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/e99ab501-cdae-42a8-b6be-bd7cf1ae33d9" />

```
df['Highly Positive Skew']=np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

```

<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/baa87bf5-8c63-43b7-a6a0-c398cdd73cc5" />

```
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df["Moderate Negative Skew"]=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```

<img width="565" height="432" alt="image" src="https://github.com/user-attachments/assets/475eefd4-682b-40a4-ad69-d6979f77fe9f" />




# RESULT:
Thus Feature Encoding and Feature Transformation has been done successfully 
