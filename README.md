# Spatial Wind Speed Forecasting Using Machine Learning

## Overview
This project uses machine learning techniques to forecast spatial wind speed in South Africa. The aim is to use historical wind speed data to create predictive models to predict future wind speed patterns.

## Related Work
The project builds upon the following research:

- **UNets**: 
  - [WF-Unet](https://arxiv.org/abs/2302.04102)

- **Transformers**: 
  - [Pangu-Weather](https://arxiv.org/abs/2211.02556) , [FengWu](https://arxiv.org/abs/2304.02948)

- **Graph Neural Networks**: 
  - [NeuralLAM](https://arxiv.org/abs/2309.17370) , [Keisler](https://arxiv.org/abs/2202.07575) , [GraphCast](https://arxiv.org/abs/2212.12794)
 
- **Multi-Layer Perceptron**:
  - Own implementation for benchmarking purposes 

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
- **Input**: The previous `n` weather states are given as images to the UNet and the forcings (hour of the day, month of the year) is also passed as input to the model.
- **Prediction Mechanism**: The PyTorch model predicts one step into the future and is utilized as an autoregressive rollout forecasting device.

## Implementations to be tested
- **Land mass forcings**: Add a layer that identifies the land mass of South Africa
- **Training for multiple steps**: In the training, increase the number of steps that the model is training.

## Model Description - Graph Convolutional Network Model
- **Encoding**: The encoder component of the GraphCast architecture maps local regions of the input into nodes of the multi-mesh graph representation (GraphCast). 
- **Processing**: The processor component updates each multi-mesh node using learned message-passing.
- **Decoding**: The decoder component maps the processed multi-mesh features back onto the grid representation.
- **Tools**: This paper shows that encoding/decoding can be done using an MLP to concatenate multiple states. 

## Issues
- **Computer crashing on PyTorch implementation**: Simulation computer with NVIDIA GeForce GPU crashing (temperature @ 73 \degrees C)
- Autoregressive rollout training not yielding the best results for UNet PyTorch Implementation
