# Time Series Exploratory Data Analysis and Forecasting - Caribbean Region Temperature Anomalies and Precipitation using XGBoost, LSTM, Prophet and SARIMAX.

## Introduction and Project Overview
This project focuses on climate variability in Caribbean Region, utilizing historical data from the National Oceanic and Atmospheric Administration (NOAA) spanning January 1980 to December 2024. The dataset includes monthly temperature anomalies (deviations from the 1910–2000 baseline) and precipitation totals, aggregated across the Caribbean basin (approximately 10–25°N, 60–85°W). Data was sourced from NOAA's Climate Prediction Center (CPC) and Physical Sciences Laboratory (PSL) portals, comprising over 900 monthly observations per variable.

The primary objectives:

- Conduct Exploratory Data Analysis (EDA) to uncover trends, seasonality, correlations, and anomalies.
- Evaluate multiple machine learning (ML) models for forecasting future temperature anomalies and precipitation (SARIMAX, XGBoost, Long Short-Term Memory (LSTM), Prophet (by Facebook))
- Forecast and simulate future scenarios Dec 2026 - Dec 2027 using the best performing ML model 
- Identify key insights for stakeholders in environmental policy, agriculture, water resource management, and disaster preparedness.

**Data source:** NOAA NCEI – Global Land Surface Temperature Anomalies + Precipitation -> [source](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/caribbeanIslands/land_ocean/tavg/1/1/1850-2025)

**Author:** Nazgul Sagatova  
**Last updated:** 2025-12-18  

## Quick Navigation (click to open each section)
1. [01-download-and-data-wrangling.ipynb](notebooks/01-download-and-data-wrangling.ipynb) – API pulls + raw files + data wrangling, missing handling, basic stats
2. [02-descriptive-stats.ipynb](notebooks/02-descriptive-stats.ipynb) – Detailed stats, skewness and kurtosis
3. [03-EDA-temperature-anomaly.ipynb](notebooks/03-EDA-temperature-anomaly.ipynb) – Main warming trends, variability, heatwaves, rolling stats, heatmap
4. [04-EDA-precipitation.ipynb](notebooks/04-EDA-precipitation.ipynb) – Rainfall trends & extremes, variability, rolling stats, heatmap
5. [05-combined-correlations.ipynb](notebooks/05-combined-correlations.ipynb) – Temp and rainfall correlations, q-q plots
6. [06-feature-engineering.ipynb](notebooks/06-feature-engineering.ipynb) - data transformation, deseasoning by LOESS, time-based features, cyclic encodings
7. [07-feature-engineering-Lags.ipynb](notebooks/07-feature-engineering-Lags.ipynb) – Lags by ACF and PACF, AIC & BIC  
8. [temperature-anomaly-forecast-xgb-arima-lstm-prophet.py](Forecasting/temperature-anomaly-forecast-xgb-arima-lstm-prophet.py) - temperature anomaly feature engineering + model training + model evaluation + forecasting future + visuals
9. [precipitation-forecast-xgb-arima-lstm-prophet.py](Forecasting/precipitation-forecast-xgb-arima-lstm-prophet.py) - precipitation feature engineering + model training + model evaluation + forecasting future + visuals
---

Preprocessing steps included identifying missing values (0%), outlier detection via the Interquartile Range (IQR) method, stationarity testing using Augmented Dickey-Fuller (ADF) tests and plotting Autocorrelation Functions (ACF and PACF) for lag features. For temperature anomalies, the ADF p-value was 0.04 (stationary after first differencing); for precipitation, it was 6.1 (requiring seasonal differencing).

Original temperature anomalies and precipitation for over 45 years (below).

<figure>
  <img width="650" height="340" alt="original_temp_precip" src="https://github.com/user-attachments/assets/6abed113-e2b2-4780-a0a3-2fe1d98ba675" />
</figure>

## Key Observations from Exploratory Data Analysis

The EDA revealed significant climate patterns influenced by global warming, ENSO cycles, and regional factors like Atlantic hurricanes.

- Caribbean has warmed **+1.4°C** since 1980 - faster than global land average
- 2024-2025 was the hottest 24-month period on record for the Caribbean region
- Years have become hotter with less fluctuations like cool nights and winters
- Years have become drier with less frequent but intense rainfall
- Overall % decrease in precipitation is 43.1% over 45 years
- By the 2020s, a month that would have been considered a 1-in-50-year extreme in the 1960s is now approaching the new normal for heavy precipitation and higher temperatures.

The heatmaps below illustrate land and sea surface temperature deviations and precipitation across Caribbean Islands and surrounding waters.

- **Temperature** Warm colors (red/orange) indicate positive anomalies (up to +1.4). The color intensifies as the anomaly grows as seen closer to 2025, reflecting sea surface temperature rise as part of global warming.
- **Precipitation** Cold colors (light blue/blue) indicate more rainfall across the islands with dark blue signifying wetter conditions (e.g. 2005 due to intense Atlantic Hurricane Season). Recent years have become drier than before.

<img width="595" height="495" alt="heatmap-both" src="https://github.com/user-attachments/assets/b348db92-1a64-4b16-972e-51c1015428bb" />

## Detailed Observations from Exploratory Data Analysis

## Temperature Anomaly
### Detecting Trends
- **Annual Trend**: Temperature anomalies show a clear **upward trend** over the decades, with the yearly mean shifting from near-zero or slightly negative values in the mid-20th century to consistently positive anomalies exceeding +0.5°C in recent years, reflecting global warming. 
- **Hottest Year**: The year **2024** recorded a mean temperature anomaly of **+1.37°C** relative to the {1910-2000} baseline, ranking as the warmest year in the 175-year record.
- **Hottest 2 Years**: The last **2 consecutive years** (2024, 2025) are the **2 warmest years** on record, confirming the ongoing long-term warming trend.
- **Hottest Decade**: In average, the temperature rose by **0.0186°C** per decade. The decade **2020** was **0.77°C** warmer than the 1980 decade, which indicates an increase of **356%**.
- **Seasonal Variations:** Summer months (Jun–Aug) show the steepest rises (+25–35% vs. 1980), linked to prolonged heatwaves and coral bleaching risks. Winter months have milder increases (+15–20%).
- **Implications:** These trends signal heightened vulnerability to extreme events; percentages highlight the non-linear acceleration, with 2020–2025 spikes exceeding +50% in peak months compared to 1980–1990 averages.

### Exploring Variability
- **No more breezy nights and cool years**: The Coefficient of Variation (CV) is decreasing over time (from **87%** in 1980s down to **27%** 2020s), the fluctuations are getting smaller relative to the rising mean anomaly. This means that natural variability in weather dominated back in 1980s with some seasons that were slightly cooler than 21st century. The year-to-year noise is very small nowadays meaning that **years have become hotter with less fluctuations**.
- **Heatwaves are frequent**: The **95th and 99th percentiles** of temperature anomalies are also rising since mid 2010, indicating that extreme heat events are becoming more frequent and intense.
- **Interannual stability**: the **10-year moving standard deviation** of temperature anomalies has remained relatively stable around 0.3–0.5°C, indicating that interannual variability has not significantly increased despite the rising mean.

## Precipitation

### Detecting Trends
- **Annual Trends**: The overall trend is going down since 1980 we can also see sudden spikes accross years. 
- **Dryness is becoming common**: Recent years starting from 2024, show significant dryness, indicated by the sudden drop at the end of the plot.
- **Atlantic Hurricane Season influence**: Years 1981, 1983, 1996, 2005 and 2010 have had the heaviest rainfall in recorded history.
- **Frequent extremes**: Mean monthly precipitation: 100 mm, with a standard deviation of 60 mm and high kurtosis (3.2), signaling frequent extreme wet/dry events.
- **September-December are wettest**: End of the year looks much wetter than the beginning of the year, with December being the wettest month, possible due to hurricane season. 
- **Almost 45% decrease in rainfall**: Overall annual avg. precipitation decrease: - 256.49mm (from 450.54mm). Overall % decrease: - -43.1% over 45 years
- **Drier, but more intense rainfall**: The number of wet months is decreasing (back in 1980s, it rained almost whole year). The intensity of rainfall is increasing, meaning that rains are becoming heavier -> 'fewer but heavier events'.

### Exploring Variability

**Heavy Rainfall/Dryness**: 
- Caribbean Region Precipitation Extremes (in months)
- Number of very dry months = 28.00
- Number of moderately extreme wet months = 28.00
- Number of very extreme wet months = 6.00

- **Consistent trend, but more extreme events**: The **12-month moving average** shows no consistent trend but strong seasonal cycles, while the **moving standard deviation** highlights elevated variability during wet seasons, with occasional spikes linked to extreme events.

## Climate is Shifting

The violin plot answers the two most important climate questions:
- Has the climate shifted? (location of the distribution)
- Has the climate become more extreme or more variable? (shape and tails of the distribution)

  
<img width="595" height="258" alt="violin-both" src="https://github.com/user-attachments/assets/1edc8a6d-9859-484b-8925-0b517455c3b8" />

**Violin Interpretation for Temperature Anomalies**
- From 1980s to 2020s, the median of temperature anomaly shifts from –0.1 to +1.1, a 1.2 z-score jump — equivalent to moving from the 46th to the 86th percentile of the 1980s distribution.
- The interquartile range (IQR) of Anomaly widens slightly (+10–15%), driven by fatter upper tails — more frequent hot outliers.

**For Caribbean temperatures** violin plots prove that global warming is not just a slow change of the average — it is a wholesale relocation of the entire temperature distribution to the warm side, with disproportionate growth in heat extremes.

**Violin Interpretation for Precipitation**
- For Caribbean precipitation, the median remains near zero (no change in typical wetness), but the 95th percentile rises dramatically (+50–100% in IQR units), confirming intensification of heavy rain.
- The lower tail (dry extremes) also extends further in recent decades, indicating more severe droughts despite stable median precipitation.

**For precipitation**, the climate has become more extreme without becoming wetter on average — the violins stay centered on zero but develop longer, fatter tails in both directions, exactly the pattern expected under anthropogenic forcing.

By the 2020s, a month that would have been considered a 1-in-50-year extreme in the 1960s is now approaching the new normal for heavy precipitation.

## Future Forecast

We employed four machine learning models in this project - SARIMAX, XGBoost, LSTM, and Prophet—to forecast monthly temperature anomalies and precipitation for the Caribbean region over a 12-month horizon (January–December 2026).

The models train on historical monthly data from January 1980 to December 2025. Prophet (ML algotirhtm by Facebook) emerges as the top performer, demonstrating the lowest out-of-sample error metrics (MAE ~0.17°C for temperature anomalies). The algorithm was trained on 515 months of temperature and precipitation data (univariately), and tested on latest 24 months. 

All models predicted the test set positively, with XGBoost being the least performing model (MAE = 0.77°C for temperature anomalies) and LSTM being the second-best. Even thought XGBoost is known for its high performance, it fails to predict the trend of time series. And this project proved it. 

| Model| Temperature MAE (°C)| Temperature RMSE (°C)| Precipitation MAE (mm) | Precipitation RMSE (mm)|
-------|---------------------|----------------------|------------------------|------------------------|
| SARIMAX | 0.36 | 0.49 | 78.3 | 95.8|
| XGBoost | 0.77 | 0.78 | 270.0 | 308.2|
| LSTM | 0.25 | 0.30 | 82.5 | 112.136|
| Prophet | 0.17 | 0.22 | 64.9 | 86.7|

## Key Forecast Insights:
 - Temperature anomalies peak in summer (Jul–Sep 2026) around +1.10–1.12°C, reflecting amplified seasonal warming. The rest of the year doesn't look much cooler though, reflecting overall global warming.
- Precipitation shows drier-than-normal conditions in (Jan-Apr 2026), with increased wet spell potential starting May towards December amid possible La Niña influences. January 2026 is forecasted with negative precipitation (-13 mm), so we expect much less rain next month.
  
<img width="1362" height="752" alt="Temp_Forecast_24" src="https://github.com/user-attachments/assets/158dc615-f419-4fee-8bac-e94f7f7b3598" />

<img width="1352" height="692" alt="Precip_Forecast_24" src="https://github.com/user-attachments/assets/322faa75-60d3-4a8c-a93e-72f809f5f1e3" />



