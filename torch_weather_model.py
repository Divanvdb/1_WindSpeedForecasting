# Third-party imports
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from datetime import datetime

import cartopy.crs as ccrs

from torch.utils.data import DataLoader

from IPython.display import HTML

import torch
import torch.nn as nn

# Local imports
from weather_data_class import WeatherData

class TorchWeatherModel(WeatherData):
    '''
    A class for training and predicting weather data using PyTorch.
    '''

    def __init__(self, ds: xr.Dataset, window_size: int = 24, steps: int = 3, use_forcings: bool = False, intervals: int = 1):
        """
        Initialize the TorchWeatherModel.

        Args:
            ds (xr.Dataset): The xarray dataset containing weather data.
            window_size (int): The size of the window for input data. Default is 24.
            steps (int): Number of future steps to predict. Default is 3.
            use_forcings (bool): Whether to use forcings such as time-based inputs (e.g., hour, month). Default is False.
            intervals (int): Interval for processing the dataset. Default is 1.
        """

        super().__init__(dataset=ds, 
                         window_size=window_size, 
                         steps=steps, 
                         auto=True, 
                         use_forcings=use_forcings, 
                         intervals=intervals)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Class setup done...')

    def assign_model(self, model: nn.Module) -> None:
        """
        Assign a model to the class instance.

        Args:
            model (nn.Module): A PyTorch model to assign for training and prediction.
        """
        self.model = model

    def train_single(self, epochs: int = 10, save_path: str = None, patience: int = 5, lr_: float = 0.0001, batch_size_: int = 128) -> None:
        """
        Train the model with a single-step training approach.

        Args:
            epochs (int): Number of training epochs. Default is 10.
            save_path (str): Path to save the best model. Default is timestamp-based.
            patience (int): Number of epochs to wait for improvement before stopping. Default is 5.
            lr_ (float): Learning rate for the optimizer. Default is 0.0001.
            batch_size_ (int): Batch size for DataLoader. Default is 128.
        """

        if save_path is None:
            save_path = f'{datetime.now().month}_{datetime.now().day}_{datetime.now().hour}_{datetime.now().minute}.pth'

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        print(f'Training on {self.device}')

        self.model.apply(self.init_weights)

        self.model.to(self.device)
        self.model.train()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, F_batch, y_batch in DataLoader(self, batch_size=batch_size_, shuffle=True):
                X_batch = X_batch.to(self.device)
                F_batch = F_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()

                if self.use_forcings:
                    outputs = self.model(X_batch, F_batch)
                else:
                    outputs = self.model(X_batch)

                loss = criterion(outputs, y_batch[:,0].reshape(-1, 1, self.dataset.latitude.size, self.dataset.longitude.size))

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(self.X_train)  # Assuming self.X_train is used for training

            scheduler.step(avg_loss)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Best Loss: {best_loss}, LR: {optimizer.param_groups[0]["lr"]}')

            # Checkpointing
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    def train_multi(self, epochs: int = 10, save_path: str = None, patience: int = 5, lr_: float = 0.0001, batch_size_: int = 128, train_steps: int = 3, load_weights: str = None) -> None:
        """
        Train the model with a multi-step autoregressive approach.

        Args:
            epochs (int): Number of training epochs. Default is 10.
            save_path (str): Path to save the best model. Default is timestamp-based.
            patience (int): Number of epochs to wait for improvement before stopping. Default is 5.
            lr_ (float): Learning rate for the optimizer. Default is 0.0001.
            batch_size_ (int): Batch size for DataLoader. Default is 128.
            train_steps (int): Number of autoregressive steps during training. Default is 3.
            load_weights (str): Path to pre-trained model weights. Default is None.
        """
        
        if save_path is None:
            save_path = f'{datetime.now().month}_{datetime.now().day}_{datetime.now().hour}_{datetime.now().minute}.pth'

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        print(f'Training on {self.device}')

        if load_weights is not None:
            self.load_model(load_weights)
        else:
            self.model.apply(self.init_weights)

        self.model.to(self.device)
        self.model.train()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, F_batch, y_batch in DataLoader(self, batch_size=batch_size_, shuffle=True):
                X_batch = X_batch.to(self.device)
                F_batch = F_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()

                cumulative_loss = 0.0
                current_input = X_batch.clone()  # Initialize predictions with the first input state
                current_F = F_batch.clone()

                # Autoregressive steps
                for step in range(train_steps):  # self.prediction_steps is how many steps you autoregress
                    
                    if self.use_forcings:
                        outputs = self.model(current_input, current_F)
                    else:
                        outputs = self.model(current_input)
                    
                    # Calculate loss at each autoregressive step
                    # print('Step' , step)
                    loss = criterion(outputs, y_batch[:, step].reshape(-1, 1, self.dataset.latitude.size, self.dataset.longitude.size))
                    cumulative_loss += loss  # Accumulate the loss

                    # Use the predicted outputs as the next inputs
                    
                    current_input = torch.cat((current_input[:, 1:], outputs), dim=1).to(self.device)

                    hour = current_F[:, 0]
                    month = current_F[:, 1]
                    
                    hour = (hour + 1) % 24

                    current_F = torch.stack((hour, month), dim=1).float().to(self.device)

                # Backpropagate the cumulative loss
                cumulative_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += cumulative_loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(self.X_train)

            scheduler.step(avg_loss)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Best Loss: {best_loss}, LR: {optimizer.param_groups[0]["lr"]}')

            # Checkpointing
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved best model to {save_path}')
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print('Early stopping triggered')
                break

    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights for the model.

        Args:
            m (nn.Module): Module whose weights will be initialized.
        """

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Conv and Linear layers
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize bias to 0

    def predict(self, X: torch.Tensor, F: torch.Tensor) -> np.ndarray:
        """
        Predict output based on input data.

        Args:
            X (torch.Tensor): Input data for prediction.
            F (torch.Tensor): Forcings data, such as hour and month (if used).

        Returns:
            np.ndarray: Model predictions.
        """

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X).float()
            F = torch.tensor(F).float()
            if self.use_forcings:
                return self.model(X, F).numpy()
            else:
                return self.model(X).numpy() # , F

    def autoregressive_predict(self, X: torch.Tensor, F: torch.Tensor, rollout_steps: int, unnormalize: bool = True, verbose: bool = False) -> np.ndarray:
        """
        Perform autoregressive predictions for multiple time steps.

        Args:
            X (torch.Tensor): Input data for prediction.
            F (torch.Tensor): Forcings data, such as hour and month.
            rollout_steps (int): Number of future steps to predict.
            unnormalize (bool): Whether to unnormalize the predictions. Default is True.
            verbose (bool): Whether to print intermediate shapes for debugging. Default is False.

        Returns:
            np.ndarray: Predictions for each time step.
        """

        self.model.eval()
        with torch.no_grad():
            
            # X = torch.tensor(X).float()
            F = torch.tensor(F).float()
            
            predictions = []

            current_input = X.to(self.device)
            current_F = F.to(self.device)
            
            for step in range(rollout_steps):
                
                if self.use_forcings:
                    next_pred = self.model(current_input, current_F).cpu().numpy()
                else:
                    try:
                        next_pred = self.model(current_input).cpu().numpy()
                    except:
                        next_pred = self.model(current_input).numpy()
                
                predictions.append(next_pred)
                
                next_pred_tensor = torch.tensor(next_pred).float().to(self.device) 

                if verbose:
                    print(current_input.shape, next_pred_tensor.shape)

                current_input = torch.cat((current_input[:, 1:], next_pred_tensor), dim=1).to(self.device)

                hour = current_F[0, 0].item()  # Extract the hour
                month = current_F[0, 1].item()  # Extract the month
                
                hour += 1
                if hour == 24:
                    hour = 0
                
                current_F = torch.tensor([[hour, month]]).float().to(self.device)

            predictions = np.array(predictions).reshape(rollout_steps, self.dataset.sizes['latitude'], self.dataset.sizes['longitude'])

            # Unnromalize the predictions
            if unnormalize:
                predictions = predictions * self.std_value + self.mean_value
            
            return predictions
        
    def save_model(self, file_path: str) -> None:
        """
        Save the current model state to a file.

        Args:
            file_path (str): Path to save the model.
        """

        torch.save(self.model.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        """
        Load a model from a file.

        Args:
            file_path (str): Path to load the model from.
        """

        self.model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def plot_pred_target(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plot the predictions and targets with animations.

        Args:
            seed (int): Seed to select the test data for plotting. Default is 0.
            frame_rate (int): Frame rate for animation. Default is 16.
            levels (int): Number of contour levels for plots. Default is 10.

        Returns:
            HTML: An HTML object containing the animation of predictions and targets.
        """

        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        targets = self.y_test[seed:seed+1]
        time_values = self.time_values

        predictions = self.autoregressive_predict(self.X_test_t[seed:seed+1], self.F_test_t[seed:seed+1], self.steps)

        fig, axs = plt.subplots(2, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(predictions.min().item(), targets.min().item())
        vmax = max(predictions.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs.flatten()[:-1]:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()

        ax_last = fig.add_subplot(2, 3, 6)

        pred = axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        error = (predictions[0] - targets[0,0].squeeze()) # / targets[0,0].squeeze() * 100
        err = axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error.squeeze(), levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        axs[0, 1].set_title('Target')
        axs[0, 0].set_title('Prediction')
        axs[0, 2].set_title('Absolute Error')

        perc_error = error / targets[0,0].squeeze() * 100
        perc_error = np.clip(perc_error, -100, 100)
        rmse = np.sqrt(error**2)

        perr = axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        rms = axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        sctr = ax_last.scatter(targets[0, 0].flatten(), predictions[0].flatten(), c=error, cmap='coolwarm')

        # plt.plot([map1.min(), map1.max()], [map1.min(), map1.max()], 'r--', label="y = x")
        

        fig.colorbar(pred, ax=axs[0, 0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[0, 1], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(err, ax=axs[0, 2], orientation='vertical', label='Percentage Error (%)')
        fig.colorbar(perr, ax=axs[1, 0], orientation='vertical', label='Percentage Error (%)')
        fig.colorbar(rms, ax=axs[1, 1], orientation='vertical', label='Root Mean Squared Error (m/s)')

        ax_last.set_xlabel("Observed Wind Speed (m/s)")
        ax_last.set_ylabel("Forecasted Wind Speed (m/s)")

        def animate(i):
            for ax in axs.flatten()[:-1]:
                ax.clear()
                ax.coastlines()
            
            ax_last.clear()
            ax_last.set_xlabel("Observed Wind Speed (m/s)")
            ax_last.set_ylabel("Forecasted Wind Speed (m/s)")

            pcm = axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[i], levels=levels, vmin=vmin, vmax = vmax)
            ptm = axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0,i % self.steps], levels=levels, vmin=vmin, vmax = vmax)
            
            error =  (predictions[i] - targets[0,i % self.steps].squeeze()) # / targets[0,i % self.steps].squeeze() * 100
            err = axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            
            perc_error = error / targets[0,i % self.steps].squeeze() * 100
            perc_error = np.clip(perc_error, -100, 100)
            rmse = np.sqrt(error**2)

            perr = axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            rms = axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            sctr = ax_last.scatter(targets[0, i % self.steps].flatten(), predictions[i].flatten(), c=error, cmap='coolwarm')

            start_time = time_values[i][0]
            end_time = time_values[i][-1]

            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)

            axs[0, 0].set_title(f'Prediction {i} - {start_time.strftime("%Y-%m-%d %H:%M:%S")} to {end_time.strftime("%Y-%m-%d %H:%M:%S")}')  
            axs[0, 1].set_title(f'Target - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            axs[0, 2].set_title(f'Error - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 0].set_title(f'Percentage Error - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 1].set_title(f'Root Mean Squared Error - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')
            ax_last.set_title(f'Error Scatter Plot - {end_time.strftime("%Y-%m-%d %H:%M:%S")}')

            # return pcm

            
        frames = predictions.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())

    def prediction_plots(self, seed: int = 0) -> None:
        """
        Generate static plots for predictions and targets.

        Args:
            seed (int): Seed to select the test data for plotting. Default is 0.
        """

        targets = self.y_test[seed:seed+1]

        predictions = self.autoregressive_predict(self.X_test_t[seed:seed+1], self.F_test_t[seed:seed+1], self.steps)

        fig, axs = plt.subplots(3, self.steps, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        # Set titles for each row
        titles_top = 'True States'
        titles_middle = 'Predictions'
        titles_bottom = 'Errors'

        # Add coastlines and titles
        for i, ax in enumerate(axs.flatten()):
            ax.coastlines()

        axs[0, self.steps // 2].set_title(f'{titles_top} @ {self.T_test[seed,0]}', fontsize=12)
        axs[1, self.steps // 2].set_title(titles_middle, fontsize=12)
        axs[2, self.steps // 2].set_title(titles_bottom, fontsize=12)

        for i in range(self.steps):
            feat = axs[0, i].contourf(self.dataset.longitude, self.dataset.latitude, targets[0, i], levels=10, transform=ccrs.PlateCarree())
            tar = axs[1, i].contourf(self.dataset.longitude, self.dataset.latitude, predictions[i], levels=10, transform=ccrs.PlateCarree())
            err = axs[2, i].contourf(self.dataset.longitude, self.dataset.latitude, predictions[i] - targets[0, i], levels=10, transform=ccrs.PlateCarree(), cmap='coolwarm')

        cbar_feat = fig.colorbar(feat, ax=axs[0, :], orientation='vertical', fraction=0.046, pad=0.04)
        cbar_tar = fig.colorbar(tar, ax=axs[1, :], orientation='vertical', fraction=0.046, pad=0.04)
        cbar_err = fig.colorbar(err, ax=axs[2, :], orientation='vertical', fraction=0.046, pad=0.04)

        plt.show()