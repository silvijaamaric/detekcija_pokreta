# Incijalizacija potrebnih biblioteka
import cv2.cv2 as cv2

import numpy as np


# Učitavanje videozapisa
cap = cv2.VideoCapture('video/video.mp4')



while cap.isOpened():

       # Čitanje stražnjih okvira (slika) iz videozapisa
       ret, frame1 = cap.read()
       print(ret)
       ret, frame2 = cap.read()
       print(ret)

       # Razlika između okvira1 (slika) i okvira2 (slika)
       diff = cv2.absdiff(frame1, frame2)

       # Pretvaranje slike u boji u sliku sive
       gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

       # Pretvaranje slike sive skale u GaussianBlur, tako da se promjene mogu lako pronaći 
       blur = cv2.GaussianBlur(gray, (5, 5), 0)       

       # Ako je vrijednost piksela veća od 20, dodjeljuje joj se bijela (255), inače crna
       _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
       dilated = cv2.dilate(thresh, None, iterations=4)

       # Pronalaženje kontura predmeta u pokretu
       contours, hirarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

       # Izrada pravokutnika oko predmeta u pokretu 
       for contour in contours:
              (x, y, w, h) = cv2.boundingRect(contour)
              if cv2.contourArea(contour) < 700:
                     continue
              cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 255), 2)

       # Prikaži izvorni okvir
       cv2.imshow('Motion Detector', frame1)

       # Prikaži diferencijalni okvir
       cv2.imshow('Difference Frame', thresh)

       # Dodijelite frame2 (slika) frame1 (image)
       frame1 = frame2

       # Pročitajte novi frame2
       ret, frame2 = cap.read()

       # Pritisnite 'esc' za izlaz 
       if cv2.waitKey(40) == 27:
              break
       if ret == False:
             print(ret)
             break
# Otpustite resurs ograničenja
cap.release()


# Uništite sve prozore
cv2.destroyAllWindows()

