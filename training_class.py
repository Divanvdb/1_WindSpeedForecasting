import torch 
import torch.nn as nn

from torch.utils.data import DataLoader

import datetime

import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation

from IPython.display import HTML

from weather_data_class_v1 import WeatherData

class TrainingClass(WeatherData):
    def __init__(self, ds: xr.Dataset, window_size: int, steps: int, use_forcings: bool = False, intervals: int = 1) -> None:
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

        # TODO: Implement the training and validation steps in the model itself to avoid code duplication

        for epoch in range(epochs):
            epoch_loss = 0
            
            # Training step
            self.data_split = 'train'
            for X_batch, F_batch, y_batch in DataLoader(self, batch_size=batch_size_, shuffle=True):
                X_batch = X_batch.to(self.device)
                F_batch = F_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()

                cumulative_loss = 0.0
                current_input = X_batch.clone()  
                current_F = F_batch.clone()

                for step in range(train_steps): 
                    
                    if self.use_forcings:
                        outputs = self.model(current_input, current_F)
                    else:
                        outputs = self.model(current_input)
                    
                    loss = criterion(outputs, y_batch[:, step].reshape(-1, 1, self.dataset.latitude.size, self.dataset.longitude.size))
                    cumulative_loss += loss  
                    
                    current_input = torch.cat((current_input[:, 1:], outputs), dim=1).to(self.device)

                    hour = current_F[:, 0]
                    month = current_F[:, 1]
                    
                    hour = (hour + 1) % 24

                    current_F = torch.stack((hour, month), dim=1).float().to(self.device)

                cumulative_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += cumulative_loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(self.X_train)

            scheduler.step(avg_loss)

            print(f'Training: Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Best Loss: {best_loss}, LR: {optimizer.param_groups[0]["lr"]}')

            # Validation step
            self.data_split = 'val'
            self.model.eval()  

            with torch.no_grad():  
                for X_batch, F_batch, y_batch in DataLoader(self, batch_size=batch_size_, shuffle=False):
                    X_batch = X_batch.to(self.device)
                    F_batch = F_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    cumulative_loss = 0.0
                    current_input = X_batch.clone()  
                    current_F = F_batch.clone()

                    for step in range(train_steps): 
                        
                        if self.use_forcings:
                            outputs = self.model(current_input, current_F)
                        else:
                            outputs = self.model(current_input)
                        
                        loss = criterion(outputs, y_batch[:, step].reshape(-1, 1, self.dataset.latitude.size, self.dataset.longitude.size))
                        cumulative_loss += loss  
                        
                        current_input = torch.cat((current_input[:, 1:], outputs), dim=1).to(self.device)

                        hour = current_F[:, 0]
                        month = current_F[:, 1]
                        
                        hour = (hour + 1) % 24

                        current_F = torch.stack((hour, month), dim=1).float().to(self.device)

                    epoch_loss += cumulative_loss.item() * X_batch.size(0)

            print(f'Validation: Epoch {epoch+1}/{epochs}, Loss: {avg_loss}, Best Loss: {best_loss}, LR: {optimizer.param_groups[0]["lr"]}')
            
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
        
        print('Training completed. Max epochs reached.')
        
    def init_weights(self, m: nn.Module) -> None:
        """
        Initialize weights for the model.

        Args:
            m (nn.Module): Module whose weights will be initialized.
        """

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) 
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def assign_model(self, model: nn.Module) -> None:
        """
        Assign a model to the class instance.

        Args:
            model (nn.Module): A PyTorch model to assign for training and prediction.
        """
        self.model = model

    def load_model(self, file_path: str) -> None:
        """
        Load a model from a file.

        Args:
            file_path (str): Path to load the model from.
        """

        self.model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

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
                return self.model(X).numpy()

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

            current_input = X#.to(self.device)
            current_F = F#.to(self.device)
            
            for step in range(rollout_steps):
                
                if self.use_forcings:
                    next_pred = self.model(current_input, current_F).cpu().numpy()
                else:
                    try:
                        next_pred = self.model(current_input).cpu().numpy()
                    except:
                        next_pred = self.model(current_input).numpy()
                
                predictions.append(next_pred)
                
                next_pred_tensor = torch.tensor(next_pred).float()#.to(self.device) 

                if verbose:
                    print(current_input.shape, next_pred_tensor.shape)

                current_input = torch.cat((current_input[:, 1:], next_pred_tensor), dim=1)#.to(self.device)

                hour = current_F[0, 0].item()  # Extract the hour
                month = current_F[0, 1].item()  # Extract the month
                
                hour += 1
                if hour == 24:
                    hour = 0
                
                current_F = torch.tensor([[hour, month]]).float()#.to(self.device)

            predictions = np.array(predictions).reshape(rollout_steps, self.dataset.sizes['latitude'], self.dataset.sizes['longitude'])

            # Unnromalize the predictions
            if unnormalize:
                predictions = predictions * self.std_value + self.mean_value
            
            return predictions
        
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
        targets = self.X_test[seed + self.window_size:seed + self.window_size + self.steps]
        time_values = self.T_test[seed + self.window_size:seed + self.window_size + self.steps]

        time_values = pd.to_datetime(time_values)

        predictions = self.autoregressive_predict(self.X_test_t[seed:seed + self.window_size].unsqueeze(0), self.F_test_t[seed + self.window_size].unsqueeze(0), self.steps)

        fig, axs = plt.subplots(2, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(predictions.min().item(), targets.min().item())
        vmax = max(predictions.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs.flatten()[:-1]:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()

        ax_last = fig.add_subplot(2, 3, 6)

        pred = axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())

        error = (predictions[0] - targets[0,0].squeeze()) 

        err = axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error.squeeze(), levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')

        perc_error = error / targets[0,0].squeeze() * 100
        perc_error = np.clip(perc_error, -100, 100)
        rmse = np.sqrt(error**2)

        perr = axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        rms = axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        ax_last.scatter(targets[0].flatten(), predictions[0].flatten(), c=error, cmap='coolwarm')

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

            axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[i], levels=levels, vmin=vmin, vmax = vmax)
            axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i], levels=levels, vmin=vmin, vmax = vmax)
            
            error =  (predictions[i] - targets[i].squeeze())
            axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            
            perc_error = error / targets[i % self.steps].squeeze() * 100
            perc_error = np.clip(perc_error, -100, 100)
            rmse = np.sqrt(error**2)

            axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            ax_last.scatter(targets[i].flatten(), predictions[i].flatten(), c=error, cmap='coolwarm')

            axs[0, 0].set_title(f'Prediction {i} - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')  
            axs[0, 1].set_title(f'Target - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[0, 2].set_title(f'Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 0].set_title(f'Percentage Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 1].set_title(f'Root Mean Squared Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            ax_last.set_title(f'Error Scatter Plot - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')

        frames = predictions.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())
