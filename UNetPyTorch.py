import torch
import torch.nn as nn

'''
Utilities to buid a Residual UNet model.

Input to the model:
    - Input weather data of shape (1, lon, lat, window_size)
    - Time input containing hour of the day and month of the year of shape (1, window_size, 2)

Output of the model:
    - Forecasted weather data of shape (1, lon, lat, 1)

'''

class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x
    
class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r(x)
        return x

class build_res_unet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        ''' Encoder 1 '''
        self.c11 = nn.Conv2d(in_c, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c, 64, kernel_size=1, padding=0) # Shortcut feature

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2)
        self.r3 = residual_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()



    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        output = self.sigmoid(output)

        return output
    
class build_res_unet_time(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        ''' Encoder 1 '''
        self.c11 = nn.Conv2d(in_c + 2, 64, kernel_size=3, padding=1)  # in_c + 2 to account for time inputs
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(in_c + 2, 64, kernel_size=1, padding=0)  # Shortcut feature

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2)
        self.r3 = residual_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, out_c, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, weather_data, time_input):
        # weather_data shape: (batch_size, window_size, lat, lon)
        # time_input shape: (batch_size, window_size, 2) where 2 corresponds to (hour of day, month of year)

        batch_size, window_size, lat, lon = weather_data.shape

        # Reshape weather_data to (batch_size, window_size, 1, lat, lon)
        weather_data = weather_data.view(batch_size, 1, lat, lon)

        # Expand time_input to match the spatial dimensions of the weather data
        time_input = time_input.view(batch_size, window_size, 2)  # (batch_size, window_size, 2)

        # Expand to match spatial dimensions (lat, lon) and channel 2 for (hour, month)
        time_input_expanded = time_input.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        time_input_expanded = time_input_expanded.expand(-1, -1, lat, lon)  # Now shape is (batch_size, 2, lat, lon)

        # Concatenate along the channel axis
        combined_input = torch.cat([weather_data, time_input_expanded], dim=1)  # New shape: (batch_size, in_c + 2, lat, lon)

        """ Encoder 1 """
        x = self.c11(combined_input)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(combined_input)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        b = self.r4(skip3)

        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ Output """
        output = self.output(d3)
        output = self.sigmoid(output)

        return output
