import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, concatenate
from sklearn.metrics import mean_squared_error

class WeatherData:
    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps = 3):
        self.dataset = dataset
        self.window_size = window_size
        self.steps = steps
        self.calculate_wind_speed()
        self.dataset = self.dataset.sortby('latitude')
    
    def subset_data(self):
        lat_slice = slice(1, 33)  
        lon_slice = slice(3, 67)  

        self.dataset = self.dataset.isel(latitude=lat_slice, longitude=lon_slice)

    def calculate_wind_speed(self):
        self.dataset['wspd'] = np.sqrt(self.dataset.u**2 + self.dataset.v**2)
        self.dataset.attrs['wspd_units'] = 'm/s'
        self.dataset['wdir'] = np.arctan2(self.dataset.v, self.dataset.u) * 180 / np.pi
        self.dataset.attrs['wdir_units'] = 'degrees'

    def window_dataset(self, variable: str = 'wspd'):
        features = []
        targets = []
        forcings = []
        time_values = []

        time_dim = self.dataset.sizes['time']
        total_windows = time_dim - self.window_size - self.steps

        for i in range(total_windows):
            print(f'{i}/{total_windows}', end='\r')
            features.append(self.dataset[variable].isel(time=slice(i, i + self.window_size)))
            targets.append(self.dataset[variable].isel(time=slice(i + self.window_size, i + self.window_size +  self.steps)))        
            time_values.append(self.dataset.time.isel(time=slice(i, i + self.window_size)).values)

            # Forcings with hour and month values
            forcings.append([self.dataset.time.isel(time=i + self.window_size).dt.hour.values, self.dataset.time.isel(time=i + self.window_size).dt.month.values])

        self.features = np.stack(features)
        self.targets = np.stack(targets)
        self.forcings = np.array(forcings)
        self.time_values = time_values

    def slice_dataset(self, end_time):
        start_time = pd.to_datetime(end_time) - pd.Timedelta(hours=self.window_size)
        return self.dataset.sel(time=slice(start_time, end_time))
    
    def weather_gifs(self, ds_, ds_f = None, feature='wspd', metric='m/s', levels=20, frames=0, frame_rate=16):

        if ds_f is None:
            vmin = ds_[feature].min().item()
            vmax = ds_[feature].max().item()
        
            fig, axs = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

            contour = ds_[feature].isel(time=0).plot.contourf(ax=axs, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)
            colorbar = plt.colorbar(contour, ax=axs)
            
        else:

            vmax = max(ds_[feature].max().values, ds_f[feature].max().values)
            vmin = min(ds_[feature].min().values, ds_f[feature].min().values)
            
            fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

            contour_f = ds_f[feature].isel(time=0).plot.contourf(ax=axs[0], levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)
            colorbar = plt.colorbar(contour_f, ax=axs[0], shrink=0.5, aspect=10)

            error = ds_[feature].values - ds_f[feature].values

            ds_error = xr.Dataset({
                feature: (('time', 'latitude', 'longitude'), error),
                'latitude': ds_.latitude,
                'longitude': ds_.longitude,
                'time': ds_.time})
            
            vmax_e = ds_error[feature].max().values
            vmin_e = ds_error[feature].min().values
            
            contour_e = ds_error[feature].isel(time=0).plot.contourf(ax=axs[2], levels=levels, vmin=vmin_e, vmax=vmax_e, add_colorbar=False, cmap='coolwarm')
            colorbar = plt.colorbar(contour_e, ax=axs[2], shrink=0.5, aspect=10)

            contour_a = ds_[feature].isel(time=0).plot.contourf(ax=axs[1], levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)
            colorbar = plt.colorbar(contour_a, ax=axs[1], shrink=0.5, aspect=10)

        def animate(i):
            if ds_f is None:
                axs.clear()  
                axs.coastlines()  
                contour = ds_[feature].isel(time=i).plot.contourf(ax=axs, levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)

                str_time = ds_.time.isel(time=i).values
                str_time = pd.to_datetime(str_time)

                axs.set_title(f'Observed {feature} {metric} at {str_time.strftime("%Y-%m-%d %H:%M:%S")} UTC')

            else:
                for ax in axs:
                    ax.clear()
                    ax.coastlines()

                contour_a = ds_[feature].isel(time=i).plot.contourf(ax=axs[1], levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)
                axs[1].set_title(f'Analysis ({feature}) {metric}')

                contour_f = ds_f[feature].isel(time=i).plot.contourf(ax=axs[0], levels=levels, vmin=vmin, vmax=vmax, add_colorbar=False)
                axs[0].set_title(f'Forecast ({feature}) {metric}')

                contour_e = ds_error[feature].isel(time=i).plot.contourf(ax=axs[2], levels=levels, vmin=vmin_e, vmax=vmax_e, add_colorbar=False, cmap='coolwarm')
                axs[2].set_title(f'Error ({feature}) {metric}')

        if frames == 0:
            frames = ds_.time.size

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

    def plot_window_target(self, seed = 0, frame_rate=16):
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


        feat = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[0], levels=20, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0], levels=20, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        axs[1].set_title('Target')

        fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')

        def animate(i):
            axs[0].clear()
            axs[0].coastlines()

            pcm = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[i], levels=20, vmin=vmin, vmax = vmax)
            

            start_time = time_values[i][0]
            end_time = time_values[i][-1]

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            axs[0].set_title(f'Window {i} - {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                ptm = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i % self.steps], levels=20, vmin=vmin, vmax = vmax)
                axs[1].set_title(f'Target - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            return pcm

            
        frames = features.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())
        
    def return_data(self):
        return self.features, self.targets, self.forcings, self.time_values


class WeatherMLModel:
    def __init__(self, model = None, steps = 3):
        """
        Initializes the WeatherMLModel class.

        Parameters:
        - model: A machine learning model (e.g., sklearn model, keras model).
        - data: The input data for training the model.
        - target: The target variable for training the model.
        """
        self.model = model
        self.features = None
        self.targets = None
        self.forcings = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.F_train = None
        self.F_test = None
        self.T_train = None
        self.T_test = None
        self.plot_shape = None
        self.predictions = None

        print('Class setup done...')

    def load_data(self, features, targets, forcings, time_values):
        """
        Loads the input data and target variable.

        Parameters:
        - features: The input data for training the model.
        - targets: The target variable for training the model.
        """

        self.plot_shape = features.shape[1:]

        self.features = features.reshape(features.shape[0], features.shape[2], features.shape[3], features.shape[1]).astype('float32')
        self.targets = targets.reshape(targets.shape[0], targets.shape[2], targets.shape[3], targets.shape[1]).astype('float32')
        self.forcings = forcings
        self.time_values = time_values

        print('Data loaded...')

    def assign_model(self, model):
        self.model = model

        print('Model assigned...')
    
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
        
    def check_model(self):
        self.model.summary()  
        
        print(self.model.predict([self.X_train[0:1], self.F_train[0:1]]).shape)
    
    def train_model(self, patience=10, best_model_name=None, max_epochs=100, val_split = 0.8, return_history=False):
        """
        Trains the machine learning model.
        """
        if best_model_name is None:
            current_time = datetime.now()
            formatted_time = current_time.strftime('%m_%d_%H_%M')

            best_model_name = f'models/{formatted_time}.h5'
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

    def plot_predictions(self, seed, levels=10, frame_rate=10):
        bounds=[15.555999755859375, 33.05699920654297, -35.13800048828125, -26.886999130249023]
        features = self.X_test[seed:seed+1]
        forcings = self.F_test[seed:seed + 1]
        targets = self.y_test[seed:seed + 1].reshape(1, self.y_test.shape[3], self.y_test.shape[1], self.y_test.shape[2])

        prediciton = self.model.predict([features, forcings]).reshape(1, self.y_test.shape[3], self.y_test.shape[1], self.y_test.shape[2])

        vmax = max(prediciton.max(), targets.max())
        vmin = min(prediciton.min(), targets.min())

        fig, axs = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        contour = axs[0].contourf(targets[seed,0], levels=levels, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(contour, ax=axs[0], shrink=0.3, aspect=10)

        error = targets - prediciton

        vmax_e = error.max()
        vmin_e = error.min()

        contour_e = axs[2].contourf(error[0,1], levels=levels, vmin=vmin_e, vmax=vmax_e, cmap='coolwarm')
        colorbar = plt.colorbar(contour_e, ax=axs[2], shrink=0.3, aspect=10)

        contour_a =axs[1].contourf(prediciton[0,0], levels=levels, vmin=vmin, vmax=vmax)
        colorbar = plt.colorbar(contour_a, ax=axs[1], shrink=0.3, aspect=10)

        def animate(i):
            for ax in axs:
                ax.clear()
                ax.coastlines()

            contour_a = axs[0].contourf(targets[0, i], levels=levels, vmin=vmin, vmax=vmax)
            axs[0].set_title(f'Analysis Wind Speed (m/s)')

            contour_f = axs[1].contourf(prediciton[0, i], levels=levels, vmin=vmin, vmax=vmax)
            axs[1].set_title(f'Forecast Wind Speed (m/s)')

            contour_e = axs[2].contourf(error[0, i], levels=levels, vmin=vmin_e, vmax=vmax_e, cmap='coolwarm')
            axs[2].set_title(f'Error Wind Speed (m/s)')

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=targets.shape[1], interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())
    