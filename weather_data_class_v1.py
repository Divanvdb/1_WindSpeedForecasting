
import xarray as xr
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cartopy.crs as ccrs

import torch.nn as nn
from torch.utils.data import Dataset

from IPython.display import HTML

import torch

from typing import Tuple

class WeatherData(Dataset):

    """
    A dataset class for preparing wind speed data for machine learning models.

    Attributes:
        dataset (xr.Dataset): The xarray dataset containing wind speed data.
        window_size (int): The size of the window for creating features.
        steps (int): The number of forecasting steps.
        use_forcings (bool): Flag to indicate whether to use forcings.
        features (np.ndarray): Array of feature data.
        targets (np.ndarray): Array of target data.
        forcings (np.ndarray): Array of forcing data.
        time_values (np.ndarray): Array of time values corresponding to features.
        min_value (float): Minimum wind speed value for normalization.
        max_value (float): Maximum wind speed value for normalization.
        mean_value (float): Mean wind speed value for normalization.
        std_value (float): Standard deviation of wind speed for normalization.
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training targets.
        y_test (np.ndarray): Testing targets.
        F_train (np.ndarray): Training forcings.
        F_test (np.ndarray): Testing forcings.
        X_train_t (torch.Tensor): Normalized training features as tensors.
        y_train_t (torch.Tensor): Normalized training targets as tensors.
        X_test_t (torch.Tensor): Normalized testing features as tensors.
        y_test_t (torch.Tensor): Normalized testing targets as tensors.
        F_train_t (torch.Tensor): Training forcings as tensors.
        F_test_t (torch.Tensor): Testing forcings as tensors.
    """

    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps: int = 3, auto: bool = True, use_forcings: bool = True, intervals: int = 1, data_split: str = 'train'):

        """
        Initializes the WeatherData object.

        Args:
            dataset (xr.Dataset): The xarray dataset containing wind speed data.
            window_size (int): The size of the window for creating features. Default is 24.
            steps (int): The number of forecasting steps. Default is 3.
            auto (bool): Flag to automatically window and normalize data. Default is False.
            use_forcings (bool): Flag to indicate whether to use forcings. Default is False.
        """
        
        self.dataset = dataset
        self.window_size = window_size
        self.steps = steps
        self.calculate_wind_speed()
        self.dataset = self.dataset.sortby('latitude')

        self.min_value = self.dataset.wspd.min().item()
        self.max_value = self.dataset.wspd.max().item()

        self.mean_value = self.dataset.wspd.mean().item()
        self.std_value = self.dataset.wspd.std().item()

        self.use_forcings = use_forcings

        self.model = None

        # MLP input size
        self.input_size = self.window_size * self.dataset.latitude.size * self.dataset.longitude.size
        self.forcing_size = 2  
        self.output_size = 1 * self.dataset.latitude.size * self.dataset.longitude.size 

        self.data_split = data_split

        if intervals > 1:
                self.time_intervals(intervals)
                

        if auto:
            self.split_data()    
            self.normalize_data()    

    def __len__(self) -> int:
        """
        Returns the number of samples based on how many windows of size `window_size + steps`
        can fit into the dataset for the specified split.

        Returns:
            int: The number of valid windows that can fit into the specified dataset split.
        """
        if self.data_split == 'train':
            dataset_length = len(self.X_train)
        elif self.data_split == 'val':
            dataset_length = len(self.X_val)
        elif self.data_split == 'test':
            dataset_length = len(self.X_test)
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")
        
        # Calculate how many windows of size `window_size + steps` fit into the dataset
        total_window_size = self.window_size + self.steps
        num_windows = dataset_length - total_window_size + 1  # Ensure correct fit
        
        return max(0, num_windows)  # Ensure the result is not negative

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the specified dataset split.

        Args:
            idx (int): The index of the sample to retrieve.
            data_split (str): The dataset split ('train', 'val', 'test'). Default is 'train'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing features, forcings, and target.
        """
        if self.data_split == 'train':
            return self.X_train_t[idx:idx+self.window_size], self.F_train_t[idx + self.window_size], self.X_train_t[idx + self.window_size:idx + self.window_size + self.steps]
        elif self.data_split == 'val':
            return self.X_val_t[idx:idx+self.window_size], self.F_val_t[idx + self.window_size], self.X_val_t[idx + self.window_size:idx + self.window_size + self.steps]
        elif self.data_split == 'test':
            return self.X_test_t[idx:idx+self.window_size], self.F_test_t[idx + self.window_size], self.X_test_t[idx + self.window_size:idx + self.window_size + self.steps]
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")

    def time_intervals(self, intervals: int = 3) -> None:
    
        """
        Subsets the dataset based on the specified time intervals. Only happens once and then dataset is saved as a .nc file.

        Args:
            intervals (int): The time intervals for subsetting. Default is 3.

        Returns:
            None: Updates the dataset in place.
        """

        self.dataset = self.dataset.sel(time=slice(None, None, intervals))

    def subset_data(self, coarsen: int = 1) -> None:

        """
        Subsets the dataset based on the specified coarsening factor. Only happens once and then dataset is saved as a .nc file.

        Args:
            coarsen (int): The coarsening factor for subsetting. Default is 1.

        Returns:
            None: Updates the dataset in place.
        """

        if coarsen > 1:
            lat_slice = slice(1, 33, coarsen)
            lon_slice = slice(3, 67, coarsen)
        else:
            lat_slice = slice(1, 33)  
            lon_slice = slice(3, 67)

        self.dataset = self.dataset.isel(latitude=lat_slice, longitude=lon_slice)

    def calculate_wind_speed(self) -> None:
        """
        Calculates wind speed from u and v components and adds it to the dataset.

        Returns:
            None: Updates the dataset in place with the wind speed variable.
        """

        self.dataset['wspd'] = np.sqrt(self.dataset.u**2 + self.dataset.v**2).astype(np.float32)
        self.dataset.attrs['wspd_units'] = 'm/s'
        # self.dataset['wdir'] = np.arctan2(self.dataset.v, self.dataset.u) * 180 / np.pi
        # self.dataset.attrs['wdir_units'] = 'degrees'

    def split_data(self, test_size: float = 0.1, val_size: float = 0.2, random_state: int = 42) -> None:
        """
        Splits the data into training, validation, and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            val_size (float): Proportion of the dataset to include in the validation split from the training set. Default is 0.1.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            None: Updates the instance attributes with training, validation, and testing sets.
        """
        
        
        # Create a numpy array of the dataset
        data = self.dataset.wspd.values
        forcings = np.stack([self.dataset.time.dt.hour.values, self.dataset.time.dt.month.values], axis=-1)
        time_values = self.dataset.time.values

        # Split the data into train, validation, and test sets

        self.X_train, self.X_test, self.F_train, self.F_test, self.T_train, self.T_test = train_test_split(data, forcings, time_values, test_size=test_size, shuffle=False)

        self.X_train, self.X_val, self.F_train, self.F_val, self.T_train, self.T_val = train_test_split(self.X_train, self.F_train, self.T_train, test_size=val_size, shuffle=False)

    def normalize_data(self, method: str = 'avg_std') -> None:
        """
        Normalizes the training, validation, and testing data using mean and standard deviation.

        Returns:
            None: Updates the instance attributes with normalized data as tensors.
        """

        self.X_train_t = self.normalize(self.X_train, method)
        self.X_val_t = self.normalize(self.X_val, method)
        self.X_test_t = self.normalize(self.X_test, method)

        # Convert to tensors
        self.X_train_t = torch.tensor(self.X_train_t).float()

        self.X_val_t = torch.tensor(self.X_val_t).float()

        self.X_test_t = torch.tensor(self.X_test_t).float()

        # Convert forcings to tensors
        self.F_train_t = torch.tensor(self.F_train).float()
        self.F_val_t = torch.tensor(self.F_val).float()
        self.F_test_t = torch.tensor(self.F_test).float()

    def normalize(self, data: np.ndarray, method: str = 'avg_std') -> np.ndarray:
        """
        Normalizes the given data using min-max normalization.

        Args:
            data (np.ndarray): The data to normalize.

        Returns:
            np.ndarray: The normalized data.
        """

        if method == 'min_max':
            return (data - self.min_value) / (self.max_value - self.min_value)
        else:
            return (data - self.mean_value) / self.std_value

    def plot_from_data(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plots features and targets from the windowed arrays for visualization.

        Args:
            seed (int): Seed for reproducibility in selecting samples. Default is 0.
            frame_rate (int): The frame rate for the animation. Default is 16.
            levels (int): Number of contour levels for the plot. Default is 10.

        Returns:
            HTML: An HTML object representing the animation.
        """
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        features = self.X_test[seed:seed + self.window_size]
        targets = self.X_test[seed + self.window_size:seed + self.window_size + self.steps]
        time_features = self.T_test[seed:seed + self.window_size]
        time_targets = self.T_test[seed + self.window_size:seed + self.window_size + self.steps]

        time_features = pd.to_datetime(time_features)
        time_targets = pd.to_datetime(time_targets)

        fig, axs = plt.subplots(1, 2, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(features.min().item(), targets.min().item())
        vmax = max(features.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()


        feat = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        axs[1].set_title('Target')

        fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')

        def animate(i):
            axs[0].clear()
            axs[0].coastlines()

            axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[i], levels=levels, vmin=vmin, vmax = vmax)

            axs[0].set_title(f'Window {i} - {time_features[i].strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i % self.steps], levels=levels, vmin=vmin, vmax = vmax)
                axs[1].set_title(f'Target - {time_targets[i % self.steps].strftime("%Y-%m-%d %H:%M:%S")}')
            # return pcm

            
        frames = features.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

    def test_class(self) -> None:
        print('self.X_train:', self.X_train.shape, 'self.X_val:', self.X_val.shape, 'self.X_test:', self.X_test.shape)
        print('self.F_train:', self.F_train.shape, 'self.F_val:', self.F_val.shape, 'self.F_test:', self.F_test.shape)

        print('self.X_train_t:', self.X_train_t.shape, 'self.X_val_t:', self.X_val_t.shape, 'self.X_test_t:', self.X_test_t.shape)
        print('self.F_train_t:', self.F_train_t.shape, 'self.F_val_t:', self.F_val_t.shape, 'self.F_test_t:', self.F_test_t.shape)

        print('self.input_size:', self.input_size, 'self.forcing_size:', self.forcing_size, 'self.output_size:', self.output_size)
