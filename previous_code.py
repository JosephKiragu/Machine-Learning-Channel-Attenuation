


# Re-preprocess the data to handle 'SYNOPCode' as a categorical variable
# Using the provided mapping for 'SYNOPCode'
SYNOP_mapping = {
    0: 'Clear',
    3: 'Dust Storm',
    4: 'Fog',
    5: 'Drizzle',
    6: 'Rain',
    7: 'Snow',
    8: 'Showers'
}

# Apply the mapping to the 'SYNOPCode' column
df['SYNOPCode'] = df['SYNOPCode'].map(SYNOP_mapping)

# Split the data into features and target variables for both models
X = df.drop(columns=['FSO_Att', 'RFL_Att'])
y_fso = df['FSO_Att']
y_rfl = df['RFL_Att']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = ['SYNOPCode']

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Preprocess the training and test sets for both models
X_train_fso, X_test_fso, y_train_fso, y_test_fso = train_test_split(X, y_fso, test_size=0.2, random_state=42)
X_train_rfl, X_test_rfl, y_train_rfl, y_test_rfl = train_test_split(X, y_rfl, test_size=0.2, random_state=42)

# Use 10% of the data for quick training
X_train_fso_preprocessed = preprocessor.fit_transform(X_train_fso[:int(0.1 * len(X_train_fso))])
X_test_fso_preprocessed = preprocessor.transform(X_test_fso[:int(0.1 * len(X_test_fso))])
X_train_rfl_preprocessed = preprocessor.fit_transform(X_train_rfl[:int(0.1 * len(X_train_rfl))])
X_test_rfl_preprocessed = preprocessor.transform(X_test_rfl[:int(0.1 * len(X_test_rfl))])

# Show the shape of the preprocessed data to confirm successful preprocessing
X_train_fso_preprocessed.shape, X_test_fso_preprocessed.shape, X_train_rfl_preprocessed.shape, X_test_rfl_preprocessed.shape


# Retrain the random forest model for 'FSO_Att' using the re-preprocessed data
# We'll use the same settings as before for quick training: 10% of the data and 10 estimators
rf_fso_retrained = RandomForestRegressor(n_estimators=10, random_state=42)
rf_fso_retrained.fit(X_train_fso_preprocessed, y_train_fso[:int(0.1 * len(y_train_fso))])

# Make predictions on a subset of the test set
y_pred_fso_retrained = rf_fso_retrained.predict(X_test_fso_preprocessed)

# Evaluate the model on the subset of the test set
rmse_fso_retrained, r2_fso_retrained = print_metrics(y_test_fso[:int(0.1 * len(y_test_fso))], y_pred_fso_retrained)

# Show RMSE and R-squared for the retrained 'FSO_Att' model
rmse_fso_retrained, r2_fso_retrained




# Retrain the random forest model for 'RFL_Att' using the re-preprocessed data
# We'll use the same settings as before for quick training: 10% of the data and 10 estimators
rf_rfl_retrained = RandomForestRegressor(n_estimators=10, random_state=42)
rf_rfl_retrained.fit(X_train_rfl_preprocessed, y_train_rfl[:int(0.1 * len(y_train_rfl))])

# Make predictions on a subset of the test set
y_pred_rfl_retrained = rf_rfl_retrained.predict(X_test_rfl_preprocessed)

# Evaluate the model on the subset of the test set
rmse_rfl_retrained, r2_rfl_retrained = print_metrics(y_test_rfl[:int(0.1 * len(y_test_rfl))], y_pred_rfl_retrained)

# Show RMSE and R-squared for the retrained 'RFL_Att' model
rmse_rfl_retrained, r2_rfl_retrained


# Perform Feature Importance Analysis for both retrained models

# Get the feature importances for 'FSO_Att'
feature_importances_fso = rf_fso_retrained.feature_importances_

# Get the feature importances for 'RFL_Att'
feature_importances_rfl = rf_rfl_retrained.feature_importances_

# Get the names of the features after one-hot encoding
feature_names = (numerical_cols +
                 [f'SYNOPCode_{category}' for category in preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[0]])

# Create dataframes for the feature importances
df_importances_fso = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_fso
}).sort_values(by='Importance', ascending=False)

df_importances_rfl = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_rfl
}).sort_values(by='Importance', ascending=False)

# Show top 5 important features for both models
df_importances_fso.head(5), df_importances_rfl.head(5)



# Build the cascading model
# For this, we'll use the already trained models for 'FSO_Att' and 'RFL_Att'

# 1. Make predictions for 'FSO_Att' on the training set
y_pred_fso_train = rf_fso_retrained.predict(X_train_fso_preprocessed)

# 2. Add these predictions as a new feature to the training set for 'RFL_Att'
X_train_rfl_cascading = np.column_stack((X_train_rfl_preprocessed, y_pred_fso_train))

# 3. Retrain the model for 'RFL_Att' using this new training set
rf_rfl_cascading = RandomForestRegressor(n_estimators=10, random_state=42)
rf_rfl_cascading.fit(X_train_rfl_cascading, y_train_rfl[:int(0.1 * len(y_train_rfl))])

# 4. Make predictions for 'FSO_Att' on the test set
y_pred_fso_test = rf_fso_retrained.predict(X_test_fso_preprocessed)

# 5. Add these predictions as a new feature to the test set for 'RFL_Att'
X_test_rfl_cascading = np.column_stack((X_test_rfl_preprocessed, y_pred_fso_test))

# 6. Make predictions for 'RFL_Att' on this new test set
y_pred_rfl_cascading = rf_rfl_cascading.predict(X_test_rfl_cascading)

# 7. Evaluate the cascading model on the subset of the test set
rmse_rfl_cascading, r2_rfl_cascading = print_metrics(y_test_rfl[:int(0.1 * len(y_test_rfl))], y_pred_rfl_cascading)

# Show RMSE and R-squared for the cascading model
rmse_rfl_cascading, r2_rfl_cascading



# Build the Multi-output Random Forest model
# We'll use the same settings as before for quick training: 10% of the data and 10 estimators

# Prepare the target variables as a multi-output array
y_train_multi = np.column_stack((y_train_fso[:int(0.1 * len(y_train_fso))], y_train_rfl[:int(0.1 * len(y_train_rfl))]))

# Train the Multi-output Random Forest model
# Note: The input features for both targets ('FSO_Att' and 'RFL_Att') are assumed to be the same for this demonstration.
rf_multi_output = RandomForestRegressor(n_estimators=10, random_state=42)
rf_multi_output.fit(X_train_fso_preprocessed, y_train_multi)

# Make predictions on a subset of the test set
y_pred_multi_output = rf_multi_output.predict(X_test_fso_preprocessed)

# Separate the predictions for 'FSO_Att' and 'RFL_Att'
y_pred_fso_multi, y_pred_rfl_multi = y_pred_multi_output[:, 0], y_pred_multi_output[:, 1]

# Evaluate the Multi-output model on the subset of the test set
rmse_fso_multi, r2_fso_multi = print_metrics(y_test_fso[:int(0.1 * len(y_test_fso))], y_pred_fso_multi)
rmse_rfl_multi, r2_rfl_multi = print_metrics(y_test_rfl[:int(0.1 * len(y_test_rfl))], y_pred_rfl_multi)

# Show RMSE and R-squared for the Multi-output Random Forest model
rmse_fso_multi, r2_fso_multi, rmse_rfl_multi, r2_rfl_multi


# It appears that the variables for the metrics of the original independent models were lost due to kernel limitations.
# For the purpose of comparison, I'll quickly re-evaluate those models on the same subset of the test set used for other models.

# Make predictions using the original independent models on a subset of the test set
y_pred_fso_original = rf_fso_retrained.predict(X_test_fso_preprocessed)
y_pred_rfl_original = rf_rfl_retrained.predict(X_test_rfl_preprocessed)

# Evaluate these predictions to get the metrics
rmse_fso_original, r2_fso_original = print_metrics(y_test_fso[:int(0.1 * len(y_test_fso))], y_pred_fso_original)
rmse_rfl_original, r2_rfl_original = print_metrics(y_test_rfl[:int(0.1 * len(y_test_rfl))], y_pred_rfl_original)

# Now create the comparison table again
comparison_data = {
    'Model Type': ['Independent (FSO_Att)', 'Independent (RFL_Att)', 'Cascading (RFL_Att)', 'Multi-output (FSO_Att)', 'Multi-output (RFL_Att)'],
    'RMSE': [rmse_fso_original, rmse_rfl_original, rmse_rfl_cascading, rmse_fso_multi, rmse_rfl_multi],
    'R-squared': [r2_fso_original, r2_rfl_original, r2_rfl_cascading, r2_fso_multi, r2_rfl_multi]
}

comparison_df = pd.DataFrame(comparison_data)

# Show the comparison summary
comparison_df


# Build the new cascading model in reverse order
# For this, we'll use the already trained models for 'FSO_Att' and 'RFL_Att'

# 1. Make predictions for 'RFL_Att' on the training set
y_pred_rfl_train = rf_rfl_retrained.predict(X_train_rfl_preprocessed)

# 2. Add these predictions as a new feature to the training set for 'FSO_Att'
X_train_fso_reverse_cascading = np.column_stack((X_train_fso_preprocessed, y_pred_rfl_train))

# 3. Retrain the model for 'FSO_Att' using this new training set
rf_fso_reverse_cascading = RandomForestRegressor(n_estimators=10, random_state=42)
rf_fso_reverse_cascading.fit(X_train_fso_reverse_cascading, y_train_fso[:int(0.1 * len(y_train_fso))])

# 4. Make predictions for 'RFL_Att' on the test set
y_pred_rfl_test = rf_rfl_retrained.predict(X_test_rfl_preprocessed)

# 5. Add these predictions as a new feature to the test set for 'FSO_Att'
X_test_fso_reverse_cascading = np.column_stack((X_test_fso_preprocessed, y_pred_rfl_test))

# 6. Make predictions for 'FSO_Att' on this new test set
y_pred_fso_reverse_cascading = rf_fso_reverse_cascading.predict(X_test_fso_reverse_cascading)

# 7. Evaluate the new cascading model on the subset of the test set
rmse_fso_reverse_cascading, r2_fso_reverse_cascading = print_metrics(y_test_fso[:int(0.1 * len(y_test_fso))], y_pred_fso_reverse_cascading)

# Show RMSE and R-squared for the new cascading model
rmse_fso_reverse_cascading, r2_fso_reverse_cascading



# Add the new cascading model to the comparison table
new_cascading_data = {
    'Model Type': ['Cascading (FSO_Att)'],
    'RMSE': [rmse_fso_reverse_cascading],
    'R-squared': [r2_fso_reverse_cascading]
}

new_cascading_df = pd.DataFrame(new_cascading_data)

# Combine the previous comparison table with this new entry
updated_comparison_df = pd.concat([comparison_df, new_cascading_df], ignore_index=True)

# Show the updated comparison summary
updated_comparison_df
