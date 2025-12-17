
import pandas as pd
import numpy as np
import requests
from pathlib import Path

def load_data():

    '''
    Function to load the precipitation data from the NOAA website:
        - loads the precipitation data
        - performs feature engineering by adding time features to the dataframe
    '''

    print("\nLoading and preparing Caribbean Region Precipitation Data...")

    
    # Download all data (tavg + pcp) in one loop. This will result in a dictionary of dataframes, 
    # each of which correspond to either precipitation or temperature anomaly

    region = 'caribbeanIslands'
    parameters={'tavg', 'pcp'}
    surface = 'land_ocean'
    timescale = 'ytd'
    month=0
    format='csv'
    begYear=1980
    endYear=2025

    skiprows = 3

    d = {name: pd.DataFrame() for name in parameters}

    for parameter in parameters:

        url = f'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/{region}/{parameter}/{surface}/{timescale}/{month}/{begYear}-{endYear}/data.{format}'

        response = requests.get(url)

        if response.status_code == 200:
            print("Success!")
            if parameter == 'pcp':
                skiprows = 2
            else:
                skiprows = 3
            print(f'skipping {skiprows} rows')
            d[parameter] = pd.read_csv(url, skiprows=skiprows)
        elif response.status_code == 404:
            print("Not Found.")


    # Merge both temperature anomaly and precipitation in one dataframe, match by date.

    print('Merging the all the data...')
    df = pd.merge(d['tavg'], d['pcp'], how = 'left', on='Date')
    print('Success on merge')

    # Convert entire column
    df['date'] = pd.to_datetime(df['Date'], format='%Y%m')

    # Set as index
    df = df.set_index('date')

    #remove the old Date format
    df.drop('Date', axis='columns', inplace=True)

    # Add seasonality features
    df['year'] = df.index.year
    df['month'] = df.index.month

    #rename the column names
    df.rename(columns={"Value" : "precip", "Anomaly" : "temp_anomaly"}, inplace=True)

    # Add log transform for precipitation
    df['precip_log'] = np.log1p(df['precip'])

    return df

def save_to_parquet(df):
    # Final save — everyone will read this file
    df.to_parquet("../data/processed/original_caribbean_temp_precip_1980_2025.parquet", index=True)
    print("Processed original dataset saved → ready")

def main():
    df = load_data()

    print(df.head())
    
    save_to_parquet(df)

if __name__ == '__main__':
    results = main()