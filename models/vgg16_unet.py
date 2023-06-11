from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import *


# VGG16 U-Net
def vgg16_encoder_unet(input_shape=(256, 256, 3), num_classes=1):
    vgg16_model = VGG16(input_shape=input_shape, weights=None, include_top=False)

    block4_conv = vgg16_model.get_layer('block5_conv3').output
    block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same')(block4_conv)
    block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same')(block5_conv1)
    block5_drop = Dropout(0.5)(block5_conv2)

    block6_up = Conv2D(512, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = Concatenate(axis=3)([vgg16_model.get_layer('block4_conv3').output, block6_up])
    block6_conv1 = Conv2D(512, 3, activation='relu', padding='same')(block6_merge)
    block6_conv2 = Conv2D(512, 3, activation='relu', padding='same')(block6_conv1)

    block7_up = Conv2D(256, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block6_conv2))
    block7_merge = Concatenate(axis=3)([vgg16_model.get_layer('block3_conv3').output, block7_up])
    block7_conv1 = Conv2D(256, 3, activation='relu', padding='same')(block7_merge)
    block7_conv2 = Conv2D(256, 3, activation='relu', padding='same')(block7_conv1)

    block8_up = Conv2D(128, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block7_conv2))
    block8_merge = Concatenate(axis=3)([vgg16_model.get_layer('block2_conv2').output, block8_up])
    block8_conv1 = Conv2D(128, 3, activation='relu', padding='same')(block8_merge)
    block8_conv2 = Conv2D(128, 3, activation='relu', padding='same')(block8_conv1)

    block9_up = Conv2D(64, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = Concatenate(axis=3)([vgg16_model.get_layer('block1_conv2').output, block9_up])
    block9_conv1 = Conv2D(64, 3, activation='relu', padding='same')(block9_merge)
    block9_conv2 = Conv2D(64, 3, activation='relu', padding='same')(block9_conv1)
    if num_classes > 1:
        block9_conv3 = Conv2D(num_classes, 1, activation='softmax')(block9_conv2)
    else:
        block9_conv3 = Conv2D(num_classes, 1, activation='sigmoid')(block9_conv2)

    model = Model(inputs=vgg16_model.input, outputs=block9_conv3)
    for index in range(41):
        model.layers[index].trainable = True
    return model
