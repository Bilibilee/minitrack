from PIL import Image
import cv2

from hdtrack.detection import TorchDetection

detection = TorchDetection()

while True:
    imgname = input('Input image filename:')
    try:
        image = Image.open(imgname)
    except:
        print('Open Error! Try again!')
        continue
    else:
        result=detection.detect_one_image(image)
        cv2.imwrite('result.jpg',result)

