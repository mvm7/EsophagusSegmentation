import cv2
import numpy as np
from PIL import Image
from time import time
from models.fcn import fcn
from models.segnet import segnet
from models.unet import unet
from models.vgg16_unet import vgg16_encoder_unet
from models.resnet101_unet import resnet101_encoder_unet
from models.vgg16_unet3plus import vgg16_encoder_unet3plus
from models.resnet101_unet3plus import resnet101_encoder_unet3plus


model = vgg16_encoder_unet3plus(input_shape=(256, 256, 3), num_classes=6)
model.load_weights('vgg16_unet3plus40.h5')


def predictMask(pilImage=None):
    if pilImage != None:
        openCVImage = np.array(pilImage)
        h, w, _ = openCVImage.shape
        openCVImage = cv2.resize(openCVImage, (256, 256), interpolation=cv2.INTER_AREA)
        openCVImage = cv2.cvtColor(openCVImage, cv2.COLOR_RGB2BGR)
        openCVImage = openCVImage / 255.0
        openCVImage = np.array([openCVImage])

        start = time()
        mask = model.predict(openCVImage, batch_size=None, verbose=0, steps=None)
        print(time()-start)
        predicted_img=np.argmax(mask, axis=3)[0,:,:]

        PILimgN = Image.new('RGBA', (256, 256), 'black')
        PILpix = PILimgN.load()
        predicted_img = np.uint8((predicted_img))
        for i in list(range(255)):
            for y in list(range(255)):
                if predicted_img[y][i]== 1:
                  PILpix[i, y] = (0, 0, 255, 255)
                if predicted_img[y][i]== 2:
                  PILpix[i, y] = (255, 0, 0, 255)
                if predicted_img[y][i]== 3:
                  PILpix[i, y] = (255, 0, 255, 255)
                if predicted_img[y][i]== 4:
                  PILpix[i, y] = (0, 255, 0, 255)
                if predicted_img[y][i]== 5:
                  PILpix[i, y] = (0, 255, 255, 255)
        mask = PILimgN.resize((w, h))
        return mask
