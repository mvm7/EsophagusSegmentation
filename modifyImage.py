# Модуль обработки изображения (сегментация, удаление фона, выделение сущностей)
import predict
from PIL import Image
from models.fcn import fcn
from models.segnet import segnet
from models.unet import unet
from models.vgg16_unet import vgg16_encoder_unet
from models.resnet101_unet import resnet101_encoder_unet
from models.vgg16_unet3plus import vgg16_encoder_unet3plus
from models.resnet101_unet3plus import resnet101_encoder_unet3plus


def getMask(img):
    return predict.predictMask(img)


def deleteBackground(img, mask):
    img = img.convert("RGBA")
    mask = mask.convert("RGBA")
    pixImg = img.load()
    pixMask = mask.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixMask[x, y] == (0, 0, 0, 255):
                pixImg[x, y] = (0, 0, 0, 0)
    return img


def highlights(img, mask):
    img = img.convert("RGBA")
    mask = mask.convert("RGBA")
    newMask = mask
    pixMask = mask.load()
    pixNewImage = newMask.load()
    width, height = mask.size
    for y in range(height):
        for x in range(width):
            if pixMask[x, y] == (255, 0, 0, 255):
                pixNewImage[x, y] = (255, 0, 0, 64)
            if pixMask[x, y] == (0, 255, 0, 255):
                pixNewImage[x, y] = (0, 255, 0, 64)
            if pixMask[x, y] == (0, 0, 255, 255):
                pixNewImage[x, y] = (0, 0, 255, 64)
            if pixMask[x, y] == (0, 255, 255, 255):
                pixNewImage[x, y] = (0, 255, 255, 64)
            if pixMask[x, y] == (255, 0, 255, 255):
                pixNewImage[x, y] = (255, 0, 255, 64)
            if pixMask[x, y] == (0, 0, 0, 255):
                pixNewImage[x, y] = (0, 0, 0, 0)
    result = Image.alpha_composite(img, newMask)
    return result