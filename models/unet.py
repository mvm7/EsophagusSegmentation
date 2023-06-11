from keras.layers import *
from keras.models import *


# U-Net
def unet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(input_shape)
    block1_conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    block1_conv2 = Conv2D(64, 3, activation='relu', padding='same')(block1_conv1)

    block2_down = MaxPooling2D(pool_size=(2, 2))(block1_conv2)
    block2_conv1 = Conv2D(128, 3, activation='relu', padding='same')(block2_down)
    block2_conv2 = Conv2D(128, 3, activation='relu', padding='same')(block2_conv1)

    block3_down = MaxPooling2D(pool_size=(2, 2))(block2_conv2)
    block3_conv1 = Conv2D(256, 3, activation='relu', padding='same')(block3_down)
    block3_conv2 = Conv2D(256, 3, activation='relu', padding='same')(block3_conv1)

    block4_down = MaxPooling2D(pool_size=(2, 2))(block3_conv2)
    block4_conv1 = Conv2D(512, 3, activation='relu', padding='same')(block4_down)
    block4_conv2 = Conv2D(512, 3, activation='relu', padding='same')(block4_conv1)

    block5_down = MaxPooling2D(pool_size=(2, 2))(block4_conv2)
    block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same')(block5_down)
    block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same')(block5_conv1)
    block5_drop = Dropout(0.5)(block5_conv2)

    block6_up = Conv2D(512, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = Concatenate(axis=3)([block4_conv2, block6_up])
    block6_conv1 = Conv2D(512, 3, activation='relu', padding='same')(block6_merge)
    block6_conv2 = Conv2D(512, 3, activation='relu', padding='same')(block6_conv1)

    block7_up = Conv2D(256, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block6_conv2))
    block7_merge = Concatenate(axis=3)([block3_conv2, block7_up])
    block7_conv1 = Conv2D(256, 3, activation='relu', padding='same')(block7_merge)
    block7_conv2 = Conv2D(256, 3, activation='relu', padding='same')(block7_conv1)

    block8_up = Conv2D(128, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block7_conv2))
    block8_merge = Concatenate(axis=3)([block2_conv2, block8_up])
    block8_conv1 = Conv2D(128, 3, activation='relu', padding='same')(block8_merge)
    block8_conv2 = Conv2D(128, 3, activation='relu', padding='same')(block8_conv1)

    block9_up = Conv2D(64, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = Concatenate(axis=3)([block1_conv2, block9_up])
    block9_conv1 = Conv2D(64, 3, activation='relu', padding='same')(block9_merge)
    block9_conv2 = Conv2D(64, 3, activation='relu', padding='same')(block9_conv1)
    if num_classes > 1:
        block9_conv3 = Conv2D(num_classes, 1, activation='softmax')(block9_conv2)
    else:
        block9_conv3 = Conv2D(num_classes, 1, activation='sigmoid')(block9_conv2)

    model = Model(inputs=inputs, outputs=block9_conv3)
    for layer in model.layers:
        layer.trainable = True
    return model
