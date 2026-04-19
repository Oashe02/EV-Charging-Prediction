import pandas as pd
import numpy as np


def preprocess_dataframe(df):
    df['time'] = pd.to_datetime(df['time'])

    df_long = df.melt(
        id_vars=['time'],
        var_name='TAZID',
        value_name='volume_kwh'
    )
    
    df_long['volume_kwh'] = df_long['volume_kwh'].fillna(0)
    df_long['volume_kwh'] = df_long['volume_kwh'].apply(lambda x: max(0, x))

    df_long['hour'] = df_long['time'].dt.hour
    df_long['day_of_week'] = df_long['time'].dt.dayofweek
    df_long['month'] = df_long['time'].dt.month
    df_long['is_weekend'] = (df_long['day_of_week'] >= 5).astype(int)

    df_long['day_of_month'] = df_long['time'].dt.day
    df_long['week_of_year'] = df_long['time'].dt.isocalendar().week.astype(int)
    df_long['quarter'] = df_long['time'].dt.quarter

    df_long['is_peak_hour'] = (
        ((df_long['hour'] >= 7) & (df_long['hour'] <= 9)) |
        ((df_long['hour'] >= 17) & (df_long['hour'] <= 20))
    ).astype(int)

    df_long['time_of_day'] = pd.cut(
        df_long['hour'],
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df_long


def get_data_summary(df_long):
    """
    generates sttaics summary
    """
    return {
        'total_records': len(df_long),
        'date_range':(
            df_long['time'].min().strftime('%Y-%m-%d'),
            df_long['time'].max().strftime('%Y-%m-%d')
        ),
        'num_zones':df_long['TAZID'].nunique(),
        'avg_volume':round(df_long['volume_kwh'].mean(),2),
        'peak_volume':round(df_long['volume_kwh'].max(),2),
        'total_volume':round(df_long['volume_kwh'].sum(),2),
    }