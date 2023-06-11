from keras.applications.resnet import ResNet101
from keras.layers import *
from keras.models import *


# ResNet101 U-Net
def resnet101_encoder_unet(input_shape=(256, 256, 3), num_classes=1):
    resnet101_model = ResNet101(input_shape=input_shape, weights=None, include_top=False)

    block4_conv = resnet101_model.get_layer('conv4_block23_out').output
    block5_conv1 = Conv2D(1024, 3, activation='relu', padding='same')(block4_conv)
    block5_conv2 = Conv2D(1024, 3, activation='relu', padding='same')(block5_conv1)
    block5_drop = Dropout(0.5)(block5_conv2)

    block6_up = Conv2D(512, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block5_drop))
    block6_merge = Concatenate(axis=3)([resnet101_model.get_layer('conv3_block4_out').output, block6_up])
    block6_conv1 = Conv2D(512, 3, activation='relu', padding='same')(block6_merge)
    block6_conv2 = Conv2D(512, 3, activation='relu', padding='same')(block6_conv1)

    block7_up = Conv2D(256, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block6_conv2))
    block7_merge = Concatenate(axis=3)([resnet101_model.get_layer('conv2_block3_out').output, block7_up])
    block7_conv1 = Conv2D(256, 3, activation='relu', padding='same')(block7_merge)
    block7_conv2 = Conv2D(256, 3, activation='relu', padding='same')(block7_conv1)

    block8_up = Conv2D(128, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block7_conv2))
    block8_merge = Concatenate(axis=3)([resnet101_model.get_layer('conv1_relu').output, block8_up])
    block8_conv1 = Conv2D(128, 3, activation='relu', padding='same')(block8_merge)
    block8_conv2 = Conv2D(128, 3, activation='relu', padding='same')(block8_conv1)

    block9_up = Conv2D(64, 2, activation='relu', padding='same')(
        UpSampling2D(size=(2, 2))(block8_conv2))
    block9_merge = Concatenate(axis=3)([resnet101_model.get_layer(index=0).output, block9_up])
    block9_conv1 = Conv2D(64, 3, activation='relu', padding='same')(block9_merge)
    block9_conv2 = Conv2D(64, 3, activation='relu', padding='same')(block9_conv1)
    if num_classes > 1:
        block9_conv3 = Conv2D(num_classes, 1, activation='softmax')(block9_conv2)
    else:
        block9_conv3 = Conv2D(num_classes, 1, activation='sigmoid')(block9_conv2)

    model = Model(inputs=resnet101_model.input, outputs=block9_conv3)
    for layer in model.layers:
        layer.trainable = True
    return model
