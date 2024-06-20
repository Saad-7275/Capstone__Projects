
---

## Walmart Sales Forecasting

This repository hosts a comprehensive analysis and forecasting project aimed at predicting weekly sales for various Walmart stores using historical data. The project applies statistical analysis, data visualization, and advanced time series forecasting techniques.

### Project Overview

- **Data Exploration**: Initial exploration of the Walmart dataset to understand trends, correlations, and distributions of sales across different stores and under varying conditions such as holidays, fuel prices, and unemployment rates.

- **Data Cleaning and Preparation**: Handling missing values and outliers to prepare the data for accurate and efficient analysis.

- **Statistical Analysis**: Detailed statistical examination including correlation analysis to identify factors influencing sales.

- **Visualization**: Utilization of libraries like Matplotlib, Seaborn, and Plotly to generate insightful visualizations such as heatmaps for correlation and box plots for outlier detection.

- **Predictive Modeling**: Implementation of time series analysis and forecasting using SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model) from the `statsmodels` library to forecast future sales. This includes dynamic forecasting to evaluate model performance over time.

### Dataset

The dataset, `Walmart Dataset.csv`, includes weekly sales from different Walmart stores along with related data such as store number, holiday flags, temperature, fuel prices, CPI (Consumer Price Index), and unemployment rates.

### Tools and Libraries Used

- **Pandas** for data manipulation.
- **Numpy** for numerical operations.
- **Matplotlib** and **Seaborn** for plotting.
- **Statsmodels** for statistical models and time series forecasting.

### Installation

Instructions to set up the required environment:

```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

### Usage

The scripts provided can be run to reproduce the analysis and the visualizations. Adjustments can be made to the store ID inputs to generate forecasts for different stores as demonstrated in the project.
---
