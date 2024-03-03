#!/usr/bin/env python
# coding: utf-8

# In[412]:


import pandas as pd


# In[413]:


import matplotlib.pyplot as plt


# In[414]:


import seaborn as sns


# In[415]:


##READ THE GIVEN DATA 


# In[416]:


data = pd.read_csv('Obesity.csv')


# In[417]:


##EXAMINING DATA 


# In[418]:


data.head() 


# In[419]:


data.info()


# In[420]:


## There are catagorical and numerical values 


# In[421]:


data.isnull().sum()


# In[422]:


##There are null values 


# In[423]:


##PREPROCESSING


# In[424]:


##Imputing null values.


# In[425]:


# Impute missing values for numerical columns with median
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in numerical_cols:
    data[col].fillna(data[col].median(), inplace=True)


# In[426]:


# Impute missing values for categorical columns with mode
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)


# In[427]:


# Verify if there are any remaining missing values
data.isnull().sum()


# In[428]:


##Identifying and Handling Outliers.


# In[429]:


# Calculate IQR for each numerical column
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1


# In[430]:


# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[431]:


# Identify outliers
outliers = ((data[numerical_cols] < lower_bound) | (data[numerical_cols] > upper_bound)).any(axis=1)


# In[432]:


# Handle outliers
data_no_outliers = data[~outliers]


# In[433]:


# Verify the shape of the new dataset
print("Shape after removing outliers:", data_no_outliers.shape)


# In[434]:


from sklearn.model_selection import train_test_split


# In[435]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder


# In[436]:


from sklearn.compose import ColumnTransformer


# In[437]:


from sklearn.pipeline import Pipeline


# In[438]:


from sklearn.linear_model import LogisticRegression


# In[439]:


from sklearn.ensemble import RandomForestClassifier


# In[440]:


from sklearn.metrics import classification_report


# In[441]:


# Split data into features and target variable


# In[442]:


X = data_no_outliers.drop('NObeyesdad', axis=1)


# In[443]:


y = data_no_outliers['NObeyesdad']


# In[444]:


# Split data into train and test sets


# In[445]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[446]:


# Define categorical and numerical columns


# In[447]:


categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']


# In[448]:


numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']


# In[449]:


# Define preprocessing steps with imputation and scaling


# In[450]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])


# In[451]:


# Define model pipelines for Logistic Regression and Random Forest
logistic_regression_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                               ('classifier', LogisticRegression(max_iter=1000))])


# In[452]:


random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('classifier', RandomForestClassifier())])


# In[453]:


# Train the Logistic Regression model
logistic_regression_pipeline.fit(X_train, y_train)


# In[454]:


# Train the Random Forest model
random_forest_pipeline.fit(X_train, y_train)


# In[455]:


##Logistic Regression and Random Forest are well-suited for predicting weight categories based on demographic information, eating habits, and physical condition. Logistic Regression provides interpretable results, making it easy to understand the impact of each feature on the prediction. While assuming linearity, it can still capture non-linear relationships through appropriate feature engineering. On the other hand, Random Forest naturally handles non-linear relationships and complex interactions in the data, making it robust and effective. Despite being sensitive to outliers, Logistic Regression is computationally efficient and performs well with linear separable data. In contrast, Random Forest's ability to handle complex datasets and high dimensionality makes it a go-to choice for classification tasks. Overall, the balance between interpretability, performance, and robustness makes Logistic Regression and Random Forest suitable options for this prediction task.


# In[456]:


##The better model has to be selected after evaluationg accuracy scores 


# In[457]:


from sklearn.metrics import accuracy_score


# In[458]:


# Calculate accuracy for Logistic Regression model
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_accuracy)


# In[459]:


# Calculate accuracy for Random Forest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)


# In[460]:


##Random forest is the superior mmodel 


# In[461]:


##User Input Prediction 


# In[468]:


# Function to preprocess user input
def preprocess_user_input(user_input):
    user_df = pd.DataFrame([user_input])
    # Impute missing numerical values with median
    user_df[numerical_cols] = user_df[numerical_cols].fillna(user_df[numerical_cols].median())
    # Impute missing categorical values with mode and one-hot encode categorical variables
    user_df[categorical_cols] = user_df[categorical_cols].fillna(user_df[categorical_cols].mode().iloc[0])
    user_df_encoded = pd.get_dummies(user_df, columns=categorical_cols)
    # Ensure the order of columns matches the order of columns in the training data
    user_df_encoded = user_df_encoded.reindex(columns=X.columns, fill_value=0)
    return user_df_encoded


# In[463]:


# Load the trained models
logistic_regression_model = logistic_regression_pipeline.named_steps['classifier']
random_forest_model = random_forest_pipeline.named_steps['classifier']


# In[464]:


# Prompt the user to input data
user_input = {}
for col in numerical_cols:
    while True:
        try:
            user_input[col] = float(input(f"Enter {col}: "))
            break
        except ValueError:
            print("Invalid input! Please enter a valid number.")

for col in categorical_cols:
    
    user_input[col] = input(f"Enter {col}: ")
    
    


# In[469]:


# Preprocess user input
user_df_encoded = preprocess_user_input(user_input)


# In[ ]:


# Make predictions
lr_prediction = logistic_regression_pipeline.predict(user_df_encoded)
rf_prediction = random_forest_pipeline.predict(user_df_encoded)


# In[ ]:


# Print the predicted weight category
print("Logistic Regression Predicted Weight Category:", lr_prediction[0])
print("Random Forest Predicted Weight Category:", rf_prediction[0])

