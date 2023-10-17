# Separate numerical and categorical features
num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove target variables from feature list
num_features.remove('FSO_Att')
num_features.remove('RFL_Att')

# Separate the data
X_num = df[num_features]
X_cat = df[cat_features]
y = df[['FSO_Att', 'RFL_Att']]

# Apply numerical transformations
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Apply categorical transformations
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)

# Combine back into a single DataFrame
X_transformed = np.hstack([X_num_scaled, X_cat_encoded])
df_transformed = pd.DataFrame(X_transformed)

# Now let's try running the cascading model function again
# Note: We'll pass the transformed X and y as a DataFrame for compatibility with the function
df_transformed = pd.DataFrame(np.hstack([X_transformed, y.values]), columns=num_features + list(X_cat_encoded.columns) + ['FSO_Att', 'RFL_Att'])


# Importing necessary libraries for data manipulation and model building
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Function to build and evaluate a cascading Random Forest Regressor model
def build_and_evaluate_cascading_rf(df, target_columns, scaler, preprocess_func=None, feature_eng_func=None, subset_size=0.1):
    """
    Build and evaluate a cascading Random Forest Regressor model.
    
    Parameters:
    df (DataFrame): The input DataFrame.
    target_columns (list): List of target columns to be predicted.
    scaler (StandardScaler): Scaler for feature scaling.
    preprocess_func (function, optional): Preprocessing function.
    feature_eng_func (function, optional): Feature engineering function.
    subset_size (float): Fraction of data to use for model building.
    
    Returns:
    dict: Dictionary containing the models and evaluation metrics.
    """
    
    # Data Preprocessing
    if preprocess_func:
        df = preprocess_func(df)
        
    # Feature Engineering
    if feature_eng_func:
        df = feature_eng_func(df)
    
    # Use a subset of the data to avoid timeout
    df_subset = df.sample(frac=subset_size, random_state=42)
    
    # Initial split for the first target
    X1 = df_subset.drop(columns=target_columns)
    y1 = df_subset[target_columns[0]]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    
    # Feature Scaling for the first target
    X1_train = scaler.fit_transform(X1_train)
    X1_test = scaler.transform(X1_test)
    
    # Model Building for the first target
    model1 = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None, 
                                   max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
    model1.fit(X1_train, y1_train)
    
    # Prediction for the first target
    y1_pred_train = model1.predict(X1_train)
    y1_pred_test = model1.predict(X1_test)
    
    # Prepare data for the second target
    X2_train = np.column_stack((X1_train, y1_pred_train))
    X2_test = np.column_stack((X1_test, y1_pred_test))
    y2 = df_subset[target_columns[1]]
    y2_train = y2[y1_train.index]
    y2_test = y2[y1_test.index]
    
    # Model Building for the second target
    model2 = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=None, 
                                   max_features='sqrt', min_samples_leaf=1, min_samples_split=2)
    model2.fit(X2_train, y2_train)
    
    # Prediction for the second target
    y2_pred = model2.predict(X2_test)
    
    # Metrics for each target
    rmse1 = np.sqrt(mean_squared_error(y1_test, y1_pred_test))
    r2_1 = r2_score(y1_test, y1_pred_test)
    rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
    r2_2 = r2_score(y2_test, y2_pred)
    
    metrics = {
        target_columns[0]: {'RMSE': rmse1, 'R2 Score': r2_1},
        target_columns[1]: {'RMSE': rmse2, 'R2 Score': r2_2}
    }
        
    return {
        'Model 1': model1,
        'Model 2': model2,
        'Metrics': metrics
    }
# Call the previously defined cascading model function
cascading_model_result = build_and_evaluate_cascading_rf(df_transformed, ['FSO_Att', 'RFL_Att'], scaler, preprocess_func=None, feature_eng_func=None, subset_size=1.0)

# Display the metrics
cascading_model_result['Metrics']
