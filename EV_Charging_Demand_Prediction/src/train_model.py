from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd



def engineer_features(df_long):
    """
    added sincos features and lag feature 

    """
    df = df_long.copy()

    df['lag_1'] = df['volume_kwh'].shift(1)
    df = df.dropna()

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    return df


def train_model(df_long, model_type="Linear Regression"):
    df_long['lag_1'] = df_long['volume_kwh'].shift(1)
    df_long = df_long.dropna()
    df_long['hour_sin'] = np.sin(2 * np.pi * df_long['hour'] / 24)
    df_long['hour_cos'] = np.cos(2 * np.pi * df_long['hour'] / 24)

    df_long['dow_sin'] = np.sin(2 * np.pi * df_long['day_of_week'] / 7)
    df_long['dow_cos'] = np.cos(2 * np.pi * df_long['day_of_week'] / 7)

    X = df_long[['hour_sin', 'hour_cos',
            'dow_sin', 'dow_cos',
                 'month', 'is_weekend','lag_1']]

    y = df_long['volume_kwh']

    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    feature_names = X.columns.tolist()

    return model, X_train, X_test, y_train, y_test, feature_names
