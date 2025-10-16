# EXNO:4-Feature Scaling and Selection


# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

## CODING AND OUTPUT:
### Feature Scaling:
<img width="1104" height="481" alt="image" src="https://github.com/user-attachments/assets/95eee06d-7a62-42f0-8a64-927b50769ba7" />
<img width="1108" height="495" alt="image" src="https://github.com/user-attachments/assets/e3642979-d951-458f-b914-4b5dfdb64fd3" />
<img width="1085" height="450" alt="image" src="https://github.com/user-attachments/assets/7b60f6d9-c497-4c3b-936c-5c09ddb87c4d" />
<img width="1076" height="460" alt="image" src="https://github.com/user-attachments/assets/e1cce522-8afc-4afe-bcb6-7f57c3d397e2" />
<img width="1082" height="469" alt="image" src="https://github.com/user-attachments/assets/a099d320-e6ac-40c3-8d6b-400cdff3b5f2" />
<img width="1081" height="470" alt="image" src="https://github.com/user-attachments/assets/38fc151a-8910-4dd7-aa8f-14d2641b7291" />

### Feature Selection:
<img width="1091" height="701" alt="image" src="https://github.com/user-attachments/assets/65f996e3-318e-4c6c-a617-edb8bc5e6bd6" />
<img width="1092" height="593" alt="image" src="https://github.com/user-attachments/assets/e8a78363-71ef-4cee-b96b-29799e1ac181" />
<img width="1094" height="276" alt="image" src="https://github.com/user-attachments/assets/82cb6532-e9a6-4ab7-8447-33622963b694" />
<img width="1088" height="437" alt="image" src="https://github.com/user-attachments/assets/0b0cdf8e-775c-4384-bfb3-20b9e88cb3a2" />
<img width="1079" height="433" alt="image" src="https://github.com/user-attachments/assets/9136ca07-d759-45b2-a0d2-460cccaaf254" />
<img width="1086" height="475" alt="image" src="https://github.com/user-attachments/assets/7aec5b4c-8282-4a33-b6f2-d9e306897971" />
<img width="1120" height="518" alt="image" src="https://github.com/user-attachments/assets/457541db-ed5a-4464-bf96-8e86c63eb707" />
<img width="1093" height="512" alt="image" src="https://github.com/user-attachments/assets/f985567c-ce62-4618-85d9-bc4587880c1d" />
<img width="1085" height="408" alt="image" src="https://github.com/user-attachments/assets/f3d81068-37bf-4246-8a27-120c62e620c6" />
<img width="1107" height="559" alt="image" src="https://github.com/user-attachments/assets/e1bc1b95-7b3f-4f0a-97a1-1cb38aec13c3" />
<img width="943" height="270" alt="image" src="https://github.com/user-attachments/assets/ef8384ec-bfe8-44c9-9b39-a14b93bd7511" />


## RESULT:
Thus, Feature selection and Feature scaling has been performed on the given dataset.
