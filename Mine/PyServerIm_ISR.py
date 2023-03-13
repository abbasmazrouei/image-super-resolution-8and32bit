import sys,socket,cv2
import numpy as np
import numpy as np
from PIL import Image  #import scipy.misc
from ISR.models import RDN, RRDN
import time
import cv2 as cv


Sock = socket.socket()

IP='127.0.0.1'
Port = 100


class PyServerIm:

    def __init__(self, ImReceiveHeight, ImReceiveWidth):
        self.ImReceive = np.zeros((ImReceiveHeight, ImReceiveWidth, 3), np.uint8)


    def ImgServer(self):
        try:
            Sock.bind((IP, Port))
            Sock.listen(5)
            print('Server is running... ')

            while True:
                Connection1, addr = Sock.accept()

                # region Code for Recieve or Send Data

                self.ServerReceive(Connection1)
                cv2.imshow("Py Receive Img", self.ImReceive)
                cv2.waitKey(1)



                # Code for ISR
                
                img1 = self.ImReceive #cv.imread('D:/A.mazrouei/A.mazrouei-Cargo/h4.tiff')

                dimensionz = img1.shape[:]
                #height=dimensionz[0]
                width=dimensionz[1]

                if (width >=2200):
                    
                    img1 = cv.resize(img1, None, fx = 1/3, fy = 1/3, interpolation = cv.INTER_CUBIC)
                    rdn = RRDN(c_dim=3,weights='gans')
                    sr_img1 = rdn.predict(img1)
                    output = cv.resize(sr_img1, None, fx = 1/2, fy = 1/4, interpolation = cv.INTER_CUBIC)
                    cv.imwrite('H4new.tiff', output)
                    
                else:
                    
                    img1 = cv.resize(img1, None, fx = 1/3, fy = 1/3, interpolation = cv.INTER_CUBIC)
                    rdn = RRDN(c_dim=3,weights='gans')
                    sr_img1 = rdn.predict(img1)
                    output = cv.resize(sr_img1, None, fx = 1/2, fy = 1/4, interpolation = cv.INTER_CUBIC)
                    cv.imwrite('H4new.tiff', output)
                
            

                # ImSend = cv2.imread("d://data/im/1.jpg")
                ImSend = self.ImReceive
                self.ServerSend(Connection1, ImSend)

                # endregion

                Connection1.close()

        except socket.error as e:
            print(e)
            return False


    def ServerReceive(self, Connection1):

        J=0
        ImLn=0
        ImByte = bytes([])
        # ImLnByte = Connection1.recv(50000)
        # ImLn = int.from_bytes(ImLnByte, sys.byteorder)

        while (len(ImByte) <= ImLn):
            ByteTmp = Connection1.recv(50000)
            if J==0:
                ImLn = int.from_bytes(ByteTmp[0:4], sys.byteorder)
                ImByte += ByteTmp[4:]
                J=1
            else:
                ImByte += ByteTmp

        decoded = np.frombuffer(ImByte[0:ImLn], dtype=np.uint8)
        self.ImReceive = decoded.reshape(self.ImReceive.shape[0], self.ImReceive.shape[1], 3)
        # self.ImReceive = decoded.reshape(ImReceive.shape[0], ImReceive.shape[1], 3)
        # cv2.imshow("Py Img 1", self.ImReceive)
        # cv2.waitKey()

    def ServerSend(self, Connection1, ImSend):

        ImByteList = self.GetImByteList(ImSend)
        NumPrt = len(ImByteList)

        # ImLnByte = ImLn.to_bytes(4, byteorder='big')
        # Connection1.send(ImLnByte)

        for i in range(NumPrt):
            Connection1.send(ImByteList[i])

    def GetImByteList(self, Im):

        ImByte = Im.tobytes()
        # global ImLn
        ImLn = len(Im.tobytes())

        # region ReconstructIm

        # decoded = np.frombuffer(ImByte[0], dtype=np.uint8)
        # # img = decoded.reshape(ImHeight, ImWidth, 1)
        # img = decoded.reshape(332, 332, 3)
        # cv2.imshow("2",img)
        # cv2.waitKey(1)

        # endregion

        NumPrt = (int)(ImLn / 8000) + 1
        ImByteList = [[0] * 8000] * NumPrt
        # ImByteList=bytearray(ImByteList)

        J = 0
        for i in range(NumPrt):

            if i==0:
                ImLnByte = ImLn.to_bytes(4, byteorder='big')
                ImByteList[i] = bytearray(ImLnByte + ImByte[0:7996])
                J+=7996
            elif i>=1 and i < NumPrt-1:
                ImByteList[i] = bytearray(ImByte[J:J + 8000])
                # print(len(ImByteList[i]))
                J += 8000
            elif i==NumPrt-1:
                ImByteList[i] = bytearray(ImByte[J:ImLn])


        return ImByteList



PS = PyServerIm(2991,2991)

PS.ImgServer()






