# Building Unet by dividing encoder and decoder into blocks

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate, add


def conv_block(input, num_filters, kernel_size=3):
    x = Conv2D(num_filters, kernel_size, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters, kernel_size_=3):
    x = conv_block(input, num_filters, kernel_size_)
    p = MaxPool2D((2, 2))(x)
    return x, p   


def decoder_block(input, skip_features, num_filters, kernel_size_=3):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, kernel_size_)
    return x


def RR_block(input, num_filters, kernel_size=3, stack_num=2, recur_num=2):
    x0 = Conv2D(num_filters, kernel_size, padding="same")(input)
    x = x0
    
    for i in range(stack_num):

        x_res = Conv2D(num_filters, kernel_size, padding="same")(x)
        x_res = BatchNormalization()(x_res)
        x_res = Activation("relu")(x_res)
            
        for j in range(recur_num):
            x_add = add([x_res, x])

            x_res = Conv2D(num_filters, kernel_size, padding="same")(x_add)
            x_res = BatchNormalization()(x_res)
            x_res = Activation("relu")(x_res)
            
        x = x_res

    x_out = add([x, x0])

    return x_out


def RR_encoder_block(input, num_filters, kernel_size_=3):
    x = RR_block(input, num_filters, kernel_size_)
    p = MaxPool2D((2, 2))(x)
    return x, p 


def RR_decoder_block(input, skip_features, num_filters, kernel_size_=3):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = RR_block(x, num_filters, kernel_size_)
    return x


def build_unet(input_shape, steps = 1):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(steps, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model


def build_RRnet(input_shape, steps = 1):
    inputs = Input(input_shape)

    s1, p1 = RR_encoder_block(inputs, 4)
    s2, p2 = RR_encoder_block(p1, 8)
    # s3, p3 = encoder_block(p2, 256)
    # s4, p4 = encoder_block(p3, 512)

    b1 = RR_block(p2, 16) #Bridge

    # d1 = decoder_block(b1, s4, 512)
    # d2 = decoder_block(d1, s3, 256)
    d3 = RR_decoder_block(b1, s2, 8)
    d4 = RR_decoder_block(d3, s1, 4)

    outputs = Conv2D(steps, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model

