# Third-party imports
import xarray as xr
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from datetime import datetime

import cartopy
import cartopy.crs as ccrs

from torch.utils.data import DataLoader, Dataset

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

    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps: int = 3, auto: bool = False, use_forcings: bool = False, intervals: int = 1, data_split: str = 'train'):

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

        # MLP input size
        self.input_size = self.window_size * self.dataset.latitude.size * self.dataset.longitude.size
        self.forcing_size = 2  
        self.output_size = 1 * self.dataset.latitude.size * self.dataset.longitude.size 

        self.data_split = data_split

        if auto:
            if intervals > 1:
                self.time_intervals(intervals)
            self.window_dataset()
            self.split_data()    
            self.normalize_data()    

    def __len__(self, ) -> int:
        """
        Returns the length of the dataset for the specified split.

        Args:
            data_split (str): The dataset split ('train', 'val', 'test'). Default is 'train'.

        Returns:
            int: The number of samples in the specified dataset split.
        """
        if self.data_split == 'train':
            return len(self.X_train)
        elif self.data_split == 'val':
            return len(self.X_val)
        elif self.data_split == 'test':
            return len(self.X_test)
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")

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
            return self.X_train_t[idx], self.F_train_t[idx], self.y_train_t[idx]
        elif self.data_split == 'val':
            return self.X_val_t[idx], self.F_val_t[idx], self.y_val_t[idx]
        elif self.data_split == 'test':
            return self.X_test_t[idx], self.F_test_t[idx], self.y_test_t[idx]
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

    def window_dataset(self, variable: str = 'wspd') -> None:

        """
        Creates windows of features and targets from the dataset.

        Args:
            variable (str): The variable to use for feature extraction. Defaults to 'wspd'.

        Returns:
            None: Updates the instance attributes with the created windows.
        """

        time_dim = self.dataset.sizes['time']
        total_windows = time_dim - self.window_size - self.steps

        # Preallocate arrays for better performance
        features = np.empty((total_windows, self.window_size, self.dataset.sizes['latitude'], self.dataset.sizes['longitude']), dtype=np.float32)
        targets = np.empty((total_windows,  self.steps, self.dataset.sizes['latitude'], self.dataset.sizes['longitude']), dtype=np.float32)
        forcings = np.empty((total_windows, 2), dtype=np.int32)
        time_values = np.empty((total_windows, self.window_size), dtype='datetime64[ns]')

        # Slice the dataset for all the time values at once
        dataset_time = self.dataset.time.values
        dataset_hour = self.dataset.time.dt.hour.values
        dataset_month = self.dataset.time.dt.month.values

        # Vectorized slicing
        for i in range(total_windows):
            print(f'{i}/{total_windows}', end='\r')
            
            # Slice features, targets, time values, and forcings in batches
            features[i] = self.dataset[variable].isel(time=slice(i, i + self.window_size)).values
            targets[i] = self.dataset[variable].isel(time=slice(i + self.window_size, i + self.window_size + self.steps)).values
            time_values[i] = dataset_time[i:i + self.window_size]

            # Hour and month forcings
            forcings[i] = [dataset_hour[i + self.window_size], dataset_month[i + self.window_size]]

        # Save arrays as attributes
        self.features = features
        self.targets = targets
        self.forcings = forcings
        self.time_values = time_values

        print('Windowed...')

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
        print('Splitting data...')

        # First, split into training and temp (validation + test)
        X_train, X_temp, y_train, y_temp, F_train, F_temp, T_train, T_temp = train_test_split(
            self.features, self.targets, self.forcings, self.time_values,
            test_size=test_size + val_size, random_state=random_state
        )

        # Next, split temp into validation and test
        val_test_ratio = val_size / (val_size + test_size)
        self.X_val, self.X_test, self.y_val, self.y_test, self.F_val, self.F_test, self.T_val, self.T_test = train_test_split(
            X_temp, y_temp, F_temp, T_temp,
            test_size=val_test_ratio, random_state=random_state
        )

        self.X_train = X_train
        self.y_train = y_train
        self.F_train = F_train
        self.T_train = T_train

        print('Shuffling training data...')
        self.X_train, self.y_train, self.F_train, self.T_train = shuffle(self.X_train, self.y_train, self.F_train, self.T_train, random_state=random_state)

    def normalize_data(self) -> None:
        """
        Normalizes the training, validation, and testing data using mean and standard deviation.

        Returns:
            None: Updates the instance attributes with normalized data as tensors.
        """
        # Normalize training data
        self.X_train_t = (self.X_train - self.mean_value) / self.std_value
        self.y_train_t = (self.y_train - self.mean_value) / self.std_value

        # Normalize validation data
        self.X_val_t = (self.X_val - self.mean_value) / self.std_value
        self.y_val_t = (self.y_val - self.mean_value) / self.std_value

        # Normalize test data
        self.X_test_t = (self.X_test - self.mean_value) / self.std_value
        self.y_test_t = (self.y_test - self.mean_value) / self.std_value

        # Convert to tensors
        self.X_train_t = torch.tensor(self.X_train_t).float()
        self.y_train_t = torch.tensor(self.y_train_t).float()

        self.X_val_t = torch.tensor(self.X_val_t).float()
        self.y_val_t = torch.tensor(self.y_val_t).float()

        self.X_test_t = torch.tensor(self.X_test_t).float()
        self.y_test_t = torch.tensor(self.y_test_t).float()

        # Convert forcings to tensors
        self.F_train_t = torch.tensor(self.F_train).float()
        self.F_val_t = torch.tensor(self.F_val).float()
        self.F_test_t = torch.tensor(self.F_test).float()

    def plot_from_ds(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plots features and targets from the dataset for visualization.

        Args:
            seed (int): Seed for reproducibility in selecting samples. Default is 0.
            frame_rate (int): The frame rate for the animation. Default is 16.
            levels (int): Number of contour levels for the plot. Default is 10.

        Returns:
            HTML: An HTML object representing the animation.
        """
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        features = self.features[seed]
        targets = self.targets[seed]
        time_values = self.time_values

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

            pcm = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[i], levels=levels, vmin=vmin, vmax = vmax)
            

            start_time = time_values[i][0]
            end_time = time_values[i][-1]

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            axs[0].set_title(f'Window {i} - {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                ptm = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i % self.steps], levels=levels, vmin=vmin, vmax = vmax)
                axs[1].set_title(f'Target - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            return pcm

            
        frames = features.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

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
        features = self.X_test[seed:seed+1]
        targets = self.y_test[seed:seed+1]
        time_values = self.time_values

        features = features * (self.max_value - self.min_value) + self.min_value
        targets = targets * (self.max_value - self.min_value) + self.min_value

        fig, axs = plt.subplots(1, 2, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(features.min().item(), targets.min().item())
        vmax = max(features.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()


        feat = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[0,0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        axs[1].set_title('Target')

        fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')

        def animate(i):
            axs[0].clear()
            axs[0].coastlines()

            pcm = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[0,i], levels=levels, vmin=vmin, vmax = vmax)
            

            start_time = time_values[i][0]
            end_time = time_values[i][-1]

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            axs[0].set_title(f'Window {i} - {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                ptm = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,i % self.steps], levels=levels, vmin=vmin, vmax = vmax)
                axs[1].set_title(f'Target - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            return pcm

            
        frames = features.shape[1]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

