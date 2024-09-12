import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.animation import FuncAnimation

from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, concatenate

from UNet import *

class WeatherData:
    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps: int = 3, auto = False):
        self.dataset = dataset
        self.window_size = window_size
        self.steps = steps
        self.calculate_wind_speed()
        self.dataset = self.dataset.sortby('latitude')

        if auto:
            self.window_dataset()
            self.split_data()        
    
    def subset_data(self, coarsen = 1):
        if coarsen > 1:
            lat_slice = slice(1, 33, coarsen)
            lon_slice = slice(3, 67, coarsen)
        else:
            lat_slice = slice(1, 33)  
            lon_slice = slice(3, 67)

        self.dataset = self.dataset.isel(latitude=lat_slice, longitude=lon_slice)

    def calculate_wind_speed(self):
        self.dataset['wspd'] = np.sqrt(self.dataset.u**2 + self.dataset.v**2).astype(np.float32)
        self.dataset.attrs['wspd_units'] = 'm/s'
        # self.dataset['wdir'] = np.arctan2(self.dataset.v, self.dataset.u) * 180 / np.pi
        # self.dataset.attrs['wdir_units'] = 'degrees'

    def window_dataset(self, variable: str = 'wspd'):
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

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the data into training, validation, and test sets.
        """

        print('Splitting...')
        self.X_train, self.X_test, self.y_train, self.y_test, self.F_train, self.F_test, self.T_train, self.T_test = train_test_split(
            self.features, self.targets, self.forcings, self.time_values,
            test_size= test_size)
     
        print('Shuffling...')
        
        self.X_train, self.y_train, self.F_train, self.T_train = shuffle(self.X_train, self.y_train, self.F_train, self.T_train, random_state=random_state)

    def plot_from_ds(self, seed = 0, frame_rate=16, levels =10):
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

    def plot_from_data(self, seed = 0, frame_rate=16, levels =10):
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        features = self.X_test[seed:seed+1]
        targets = self.y_test[seed:seed+1]
        time_values = self.time_values

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

    def return_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test, self.F_train, self.F_test, self.T_train, self.T_test

class WeatherMLModel(WeatherData):
    def __init__(self, ds=None, window_size=3, steps=3):
        """
        Initializes the WeatherMLModel class.

        Parameters:
        - model: A machine learning model (e.g., sklearn model, keras model).
        - data: The input data for training the model.
        - target: The target variable for training the model.
        """
        super().__init__(dataset=ds, window_size=window_size, steps=steps, auto=True)

        self.prep_data()

        print('Class setup done...')

    def prep_data(self):
        """
        Converts the numpy arrays to tensors and reshapes the data.
        """

        # Reshape the data
        self.X_train_tensor = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2], self.X_train.shape[3], self.X_train.shape[1])
        self.X_test_tensor = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2], self.X_test.shape[3], self.X_test.shape[1])

        self.y_train_tensor = self.y_train.reshape(self.y_train.shape[0], self.y_train.shape[2], self.y_train.shape[3], self.y_train.shape[1])
        self.y_test_tensor = self.y_test.reshape(self.y_test.shape[0], self.y_test.shape[2], self.y_test.shape[3], self.y_test.shape[1])

        # To tensor values for the model

        self.X_train_tensor = tf.convert_to_tensor(self.X_train_tensor, dtype=tf.float32)
        self.X_test_tensor = tf.convert_to_tensor(self.X_test_tensor, dtype=tf.float32)

        self.y_test_tensor = tf.convert_to_tensor(self.y_test_tensor, dtype=tf.float32)
        self.y_train_tensor = tf.convert_to_tensor(self.y_train_tensor, dtype=tf.float32)

        self.F_train_tensor = tf.convert_to_tensor(self.F_train, dtype=tf.float32)
        self.F_test_tensor = tf.convert_to_tensor(self.F_test, dtype=tf.float32)

        print('Data prepared...')

    def assign_model(self, model):
        self.model = model

        print('Model assigned...')

    def check_model(self):
        self.model.summary()  

        print(self.model.predict([self.X_train_tensor[0:1], self.F_train_tensor[0:1]]).shape)
    
    def train_model(self, patience=10, best_model_name=None, max_epochs=100, val_split = 0.8, return_history=False):
        """
        Trains the machine learning model.
        """
        if best_model_name is None:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%m_%d_%M')

            best_model_name = f'{formatted_time}.h5'
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min', verbose=1)
  
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                best_model_name,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0
            )
        
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
        
        print('Compiled...')

        if val_split != 0:
            split = int(self.X_train.shape[0] * val_split)

            history = self.model.fit([self.X_train[:split], self.F_train[:split]], self.y_train[:split], epochs=max_epochs,
                        validation_data=([self.X_train[split:], self.F_train[split:]], self.y_train[split:]),
                        callbacks=[early_stopping, model_checkpoint])
        else:
            history = self.model.fit([self.X_train, self.F_train], self.y_train, epochs=max_epochs,
                        callbacks=[early_stopping, model_checkpoint])
            
        if return_history:
            return history

    def evaluate_model(self):
        """
        Evaluates the trained model.
        """
        self.predictions = self.model.predict([self.X_test, self.F_test])

        return mean_squared_error(self.y_test.flatten(), self.predictions.flatten(), squared=False)

    def load_model(self, filepath):
        """
        Loads a model from a file.

        Parameters:
        - filepath: The path to the file from which the model will be loaded.
        """
        self.model = tf.keras.models.load_model(filepath)

    def plot_from_tensor(self, seed = 0, frame_rate=16, levels =10):
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        features = self.X_test_tensor[seed:seed+1].numpy().reshape(1, self.X_test.shape[1], self.X_test.shape[2], self.X_test.shape[3])
        targets = self.y_test_tensor[seed:seed+1].numpy().reshape(1, self.y_test.shape[1], self.y_test.shape[2], self.y_test.shape[3])
        time_values = self.time_values

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
    
    def plot_predictions(self, seed = 0, frame_rate=16, levels =10):
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        features = self.X_test_tensor[seed:seed+1]
        forcings = self.F_test_tensor[seed:seed+1]
        targets = self.y_test_tensor[seed:seed+1].numpy().reshape(1, self.y_test.shape[1], self.y_test.shape[2], self.y_test.shape[3])
        time_values = self.time_values

        predictions = self.model.predict([features, forcings]).reshape(1, self.y_test.shape[1], self.y_test.shape[2], self.y_test.shape[3])

        error = targets - predictions

        fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        amin = targets.min()
        amax = targets.max()

        pmin = predictions.min()
        pmax = predictions.max()

        emin = error.min()
        emax = error.max()

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()

        feat = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[0,0], levels=levels, vmin=pmin, vmax = pmax, transform=ccrs.PlateCarree())
        tar = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,0], levels=levels, vmin=amin, vmax = amax, transform=ccrs.PlateCarree())
        err = axs[2].contourf(self.dataset.longitude, self.dataset.latitude, error[0,0], levels=levels, vmin=emin, vmax = emax, transform=ccrs.PlateCarree())
        axs[1].set_title('Target')
        axs[2].set_title('Error')

        fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(err, ax=axs[2], orientation='vertical', label='Error (m/s)')

        def animate(i):
            axs[0].clear()
            axs[0].coastlines()

            pcm = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[0,i], levels=levels, vmin=pmin, vmax = pmax)
            
            start_time = time_values[i][0]
            end_time = time_values[i][-1]

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            axs[0].set_title(f'Predictions {i} - {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                ptm = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,i % self.steps], levels=levels, vmin=amin, vmax = amax)
                axs[1].set_title(f'Target - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')

                err = axs[2].contourf(self.dataset.longitude, self.dataset.latitude, error[0,i % self.steps], levels=levels, vmin=error.min(), vmax = error.max())
                axs[2].set_title(f'Error - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            return pcm

            
        frames = targets.shape[1]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

