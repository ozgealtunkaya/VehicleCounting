import cv2
import numpy as np

video = cv2.VideoCapture("yol_durumu.mp4")

ret, frame = video.read()

fgbg = cv2.createBackgroundSubtractorMOG2()

kucuk_video = frame[0:450, 0:450]

kernel = np.ones((5,5),np.uint8)

class Koordinat:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class Sensor:
    def __init__(self, koordinat1, koordinat2, genislik, uzunluk):
        self.koordinat1 = koordinat1
        self.koordinat2 = koordinat2
        self.genislik = genislik
        self.uzunluk = uzunluk
        self.Maske = np.zeros((genislik, uzunluk,1),np.uint8)*abs(self.koordinat2.y-self.koordinat1.y)
        self.maske_alani = abs(self.koordinat2.x-self.koordinat1.x)
        cv2.rectangle(self.Maske,(self.koordinat1.x,self.koordinat1.y),(self.koordinat2.x,self.koordinat2.y),(255),thickness=cv2.FILLED)
        self.Durum = False
        self.Arac_Sayisi = 0

sensor_var = Sensor(Koordinat(1, kucuk_video.shape[1] - 35), Koordinat(440, kucuk_video.shape[1] - 30), kucuk_video.shape[0], kucuk_video.shape[1])

font = cv2.FONT_HERSHEY_TRIPLEX

while (1):
    ret,frame = video.read()
    kucuk_video = frame[0:450, 0:450]
    sade_video = fgbg.apply(kucuk_video)
    sade_video = cv2.morphologyEx(sade_video,cv2.MORPH_OPEN,kernel)

    cnts,_ = cv2.findContours(sade_video,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    Goruntu = kucuk_video.copy()

    kutu = np.zeros((kucuk_video.shape[0], kucuk_video.shape[1], 1), np.uint8)

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if (75<w<160 and 75<h<160):
            cv2.rectangle(Goruntu, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
            cv2.rectangle(kutu, (x, y), (x+w, y+h), (255), thickness=cv2.FILLED)

    Maske1 = np.zeros((kutu.shape[0], kutu.shape[1], 1), np.uint8)
    Maske1Sonuc = cv2.bitwise_or(kutu, kutu, mask=sensor_var.Maske)
    beyaz_piksel = np.sum(Maske1Sonuc == 255)
    sensor_alani = beyaz_piksel/sensor_var.maske_alani
    if sensor_alani > 0:
        print("sensor_alani : ", sensor_alani)

    if (sensor_alani>=0.9 and sensor_var.Durum==False):
        cv2.rectangle(Goruntu, (sensor_var.koordinat1.x, sensor_var.koordinat1.y),
                      (sensor_var.koordinat2.x, sensor_var.koordinat2.y),
                      (0, 255, 0,), thickness=cv2.FILLED)
        sensor_var.Durum = True
    elif (sensor_alani<0.9 and sensor_var.Durum==True) :
        cv2.rectangle(Goruntu, (sensor_var.koordinat1.x, sensor_var.koordinat1.y),
                      (sensor_var.koordinat2.x, sensor_var.koordinat2.y),
                      (0, 0,255), thickness=cv2.FILLED)
        sensor_var.Durum = False
        sensor_var.Arac_Sayisi+=1
    else:
        cv2.rectangle(Goruntu, (sensor_var.koordinat1.x, sensor_var.koordinat1.y),
                      (sensor_var.koordinat2.x, sensor_var.koordinat2.y),
                      (0, 0, 255), thickness=cv2.FILLED)

    cv2.putText(Goruntu,str(sensor_var.Arac_Sayisi),(sensor_var.koordinat1.x,120),font,2,(0,255,255))
    cv2.imshow("video", Goruntu)
    #cv2.imshow("kutu", kutu)

    k = cv2.waitKey(30) & 0xff
    if k == 27 :
        break

video.release()
cv2.destroyAllWindows()