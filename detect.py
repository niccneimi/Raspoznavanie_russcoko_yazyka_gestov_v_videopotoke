from time import sleep
from tensorflow.keras.preprocessing import image
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model

labels=["NONE","А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц", "Ч", "Ш", "Ы", "Ь", "Э", "Ю", "Я"]
wn = cv2.VideoCapture(0)#захват изображения с камеры номер 0
detector = HandDetector(maxHands=1)#параметры для поиска 1ой руки на изображении
classifier = load_model('Trained_model_96proz.h5')
out = 20
fixsize=300
dsize = (400, 400)
folder = 'Data/29'
test_img_folder = 'Data_test/29'
counter=0

cam = cv2.VideoCapture(0)

while True:
    success, img = wn.read()#обработка полученных данных с камеры
    imgOutput =img.copy()
    hands = detector.findHands(img, draw=False)#захват руки на выводе по параметрам detector
    
    #обрезаем руку из основного изображения и создание базы изображения для обучения
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgforteach= np.ones((fixsize,fixsize,3),np.uint8)#создание квадратного изображения
        imgcut = img[y-out:y+h+out,x-out:x+w+out]#строка регулирующая степень обрезки
        imgcutshape=imgcut.shape

        ratio = h/w
        try:
            if ratio>1:#растягиваем изображение по высоте и центрируем
                k = fixsize/h
                wcal = math.ceil(k*w)
                resize = cv2.resize(imgcut,(wcal,fixsize))#параметр растяжения
                imgresizeshape=resize.shape
                wgap = math.ceil((fixsize-wcal)/2)#параметр перемещения в центр
                imgforteach[:, wgap:wcal+wgap]=resize#перемещение imgcut на imgforteach
                bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
                fgmask = bgModel.apply(imgforteach, learningRate=0)
                kernel = np.ones((3, 3), np.uint8)
                fgmask = cv2.erode(fgmask, kernel, iterations=1)
                res = cv2.bitwise_and(imgforteach, imgforteach, mask=fgmask)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (41, 41), 0)
                ret, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.resize(thresh, (64,64))
                threshed=image.img_to_array(thresh)
                threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2RGB)
                threshed=np.expand_dims(threshed,axis=0)
                prediction=classifier.predict(threshed)
                index = np.argmax(prediction)
                output = cv2.resize(thresh, dsize)

            else:#растягиваем по ширине и центрируем
                k = fixsize/w
                hcal = math.ceil(k*h)
                resize = cv2.resize(imgcut,(fixsize,hcal))#параметр растяжения
                imgresizeshape=resize.shape
                hgap = math.ceil((fixsize-hcal)/2)#параметр перемещения в центр
                imgforteach[hgap:hcal+hgap, :]=resize#перемещение imgcut на imgforteach
                bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
                fgmask = bgModel.apply(imgforteach, learningRate=0)
                kernel = np.ones((3, 3), np.uint8)
                fgmask = cv2.erode(fgmask, kernel, iterations=1)
                res = cv2.bitwise_and(imgforteach, imgforteach, mask=fgmask)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (41, 41), 0)
                ret, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thresh = cv2.resize(thresh, (64,64))
                threshed=image.img_to_array(thresh)
                threshed = cv2.cvtColor(threshed, cv2.COLOR_BGR2RGB)
                threshed=np.expand_dims(threshed,axis=0)
                prediction=classifier.predict(threshed)
                index = np.argmax(prediction)
                output = cv2.resize(thresh, dsize)
            cv2.putText(img,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        except cv2.error:
            while cv2.error:
                cv2.destroyWindow('Cameracut')
                print("### !!!ОТОДВИНЬТЕ РУКУ ДАЛЬШЕ!!!")
                break
            imgcut = img[:,:]
            cv2.imshow("Cameracut", imgOutput)


        cv2.imshow("Cameracut", imgOutput)
        cv2.imshow("Camerateach", output)

    
    #Вывод изображения с камеры
    cv2.imshow("Camera", img)
    key = cv2.waitKey(1)
    #сохраняем изображения для обучения в папки
    if key == ord("s") and counter<300:
        counter+=1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg',imgforteach)
        print(counter)
    if key == ord("s") and counter>=300 and counter<400:
        counter+=1
        cv2.imwrite(f'{test_img_folder}/Image_{counter}.jpg',imgforteach)
        print(counter)
    
