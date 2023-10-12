
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('RFLFSODataFull.csv')

# Data Preprocessing
def preprocess_data(df):
    SYNOP_mapping = {
        0: 'Clear',
        3: 'Dust Storm',
        4: 'Fog',
        5: 'Drizzle',
        6: 'Rain',
        7: 'Snow',
        8: 'Showers'
    }
    df['SYNOPCode'] = df['SYNOPCode'].map(SYNOP_mapping)
    df = pd.get_dummies(df, columns=['SYNOPCode'], drop_first=True)
    return df

df = preprocess_data(df)

# Feature Scaling
scaler = StandardScaler()

# Build and evaluate individual Random Forest models for FSO_Att and RFL_Att
# ... (Similar to code executed in the chat)

# Build and evaluate Cascading Random Forest model
# ... (Similar to code executed in the chat)

# Build and evaluate Multi-Output Random Forest model
# ... (Similar to code executed in the chat)

# Build and evaluate Reverse Cascading Random Forest model
# ... (Similar to code executed in the chat)

# Metrics evaluation (RMSE, R2)
# ... (Similar to code executed in the chat)

# Add your own code for further analysis or model deployment
