import pickle
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn import model_selection as cv
pic = open('handgesturesvmmodel.pickle','rb')
model=pickle.load(pic)
import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret1, frame = cap.read()
    else:
        ret1 = False
    while ret1:

        ret, frame = cap.read()
        img=frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        th = 160
        max_val = 255
        ret, o1 = cv2.threshold(img, th, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, o2 = cv2.threshold(img, th, max_val, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ret, o3 = cv2.threshold(img, th, max_val, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
        ret, o4 = cv2.threshold(img, th, max_val, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
        ret, o5 = cv2.threshold(img, th, max_val, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

        o2=cv2.medianBlur(o2,7)

        cv2.rectangle(img, (30, 30), (300, 300), (0, 255, 0), 2)
        o2=o2[30:300,30:300]

        o2 = cv2.resize(o2, None, fx=0.2666, fy=0.2666, interpolation=cv2.INTER_AREA)

        output = [img, o1, o2, o3, o4, o5]

        titles = ['Original', 'Binary', 'Binary Inv',
                  'Zero', 'Zero Inv', 'Trunc']


        #cv2.imshow(titles[1], output[1])
        cv2.imshow(titles[2], output[2])
        #cv2.imshow(titles[3], output[3])
        #cv2.imshow(titles[4], output[4])
        #cv2.imshow(titles[5], output[5])
        #print(o2)
        #imga, g, r = cv2.split(o2)
        text=['','five','fist','close']
        imga = o2.ravel()

        imga = np.where(imga > 0, 1, 0)
        p=model.predict([imga])
        print(int(p))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text[int(p)], (250, 250), font, 4, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(titles[0], output[0])





        if cv2.waitKey(1) == 27:  # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()