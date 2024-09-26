# Spatial Wind Speed Forecasting Using Machine Learning

## Overview
This project focuses on forecasting spatial wind speed in South Africa using machine learning techniques. The aim is to utilize historical wind speed data to create predictive models that can forecast future wind speed patterns.

## Related Work
The project builds upon the following research:

- **UNets**: 
  - [WF-Unet](https://arxiv.org/abs/2302.04102)

- **Transformers**: 
  - [Pangu-Weather](https://arxiv.org/abs/2211.02556) 
  - [FengWu](https://arxiv.org/abs/2304.02948)

- **Graph Neural Networks**: 
  - [NeuralLAM](https://arxiv.org/abs/2309.17370) 
  - [Keisler](https://arxiv.org/abs/2202.07575) 
  - [GraphCast](https://arxiv.org/abs/2212.12794)

## Data Description
- **Source**: ERA5 wind speed data for South Africa (2018-2022).
- **Format**: Data is stored as a .nc file and opened using xarray.
- **Shape**: The dataset has the following shape: `[time_steps, latitude, longitude, wind_speed_values]`.

## Data Processing
1. **Windowing**: The data is windowed based on a specified `window_size` and `step_size` to create features, forcings, and targets for the machine learning models.
   
2. **Feature Extraction**:
   - **Features**: Previous `n` steps based on the `window_size`.
   - **Forcings**: Current hour of the day and month of the year at the time of prediction.
   - **Target**: The `m` number of forecasting steps based on the defined `steps` variable.

## Model Description - UNet Model
- **Current Model**: A UNet model that integrates the feature states and the forcings as input.
- **Prediction Mechanism**: The PyTorch model predicts one step into the future and is utilized as an autoregressive rollout forecasting device.

## Implementations to be tested
- **Land mass forcings**: Add a layer that identifies the land mass of South Africa
- **Training for multiple steps**: In the training, increase the number of steps that the model is training.

## Issues
- **Computer crashing on PyTorch implementation**: Simulation computer with NVIDIA GeForce GPU crashing (temperature @ 73 \degrees C)
