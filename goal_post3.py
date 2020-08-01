import cv2
import numpy as np
import math

#Define method for calculate absoulute length of the line
def lengthOfLine(x1,y1,x2,y2):
    length=math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return length

#Reading the video
vidcap = cv2.VideoCapture('cut1.mp4')
success,img = vidcap.read()
maxLine = 0
kernel1 = np.ones((4,4),np.uint8)
kernel2 = np.ones((3,3),np.uint8)

#calculate width and hieght of image frame in number of pixels
dimentions=img.shape
hieght=dimentions[0]
width=dimentions[1]

count = 0
success = True
idx = 0
lower_white = np.array([200,200,200])
upper_white = np.array([255,255,255])

#Read the video frame by frame
while success:

    # img =cv2.imread('test_image2.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,blackCopy=cv2.threshold(gray,255,255,cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(img,lower_white,upper_white)
    maskedImg=cv2.bitwise_and(img,img,mask=mask)

    bgrImg=cv2.cvtColor(maskedImg,cv2.COLOR_HSV2BGR)
    grayImg=cv2.cvtColor(bgrImg,cv2.COLOR_BGR2GRAY)

    img_with_edges=cv2.Canny(grayImg,50,150)
    dilatedImg = cv2.dilate(img_with_edges,kernel1,iterations=1)
    erosionedImg=cv2.erode(dilatedImg,kernel2,iterations=1)
    cv2.imwrite("./Dilated_img/frame%d.jpg"%count,dilatedImg)

    #vertical lines
    img_with_edges_rgb=cv2.cvtColor(dilatedImg,cv2.COLOR_GRAY2BGR)
    minLineLength = 5
    maxLineGap = 10
    linesP = cv2.HoughLinesP(dilatedImg,1,np.pi/2,5,minLineLength,maxLineGap)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            # print(l)
            linelength=lengthOfLine(l[0],l[1],l[2],l[3])
            if ((l[0]==l[2])and(0.1*hieght<=linelength<=0.75*hieght)):
                cv2.line(blackCopy, (l[0], l[1]), (l[2], l[3]), (255,255,255)) #write all vertical lines in blackCopy image
                cv2.imwrite("./Blackcopy/frame%d.jpg"%count,blackCopy)

    linesOfblackCopy=cv2.HoughLinesP(blackCopy,1,np.pi/2,5,minLineLength,maxLineGap)
    #calculate the maxline length
    if linesOfblackCopy is not None:
        for i in range(0,len(linesOfblackCopy)):
            l = linesOfblackCopy[i][0]
            lineLength = lengthOfLine(l[0],l[1],l[2],l[3])
            if (lineLength>=maxLine):
                maxLine = lineLength

    #distance between two vertical lines condition checking
    if linesOfblackCopy is not None:
        if len(linesOfblackCopy)>=2:
            for i in range(0,len(linesOfblackCopy)):
                l1=linesOfblackCopy[i][0]
                for j in range(i+1,len(linesOfblackCopy)):
                    l2=linesOfblackCopy[j][0]
                    if l1 is not l2:
                        distant=abs(l1[0]-l2[0])
                        lengthOfl1=lengthOfLine(l1[0],l1[1],l1[2],l1[3])
                        lengthOfl2=lengthOfLine(l2[0],l2[1],l2[2],l2[3])
                        # maxline=max([lengthOfl1,lengthOfl2])
                        print('distance between lines',distant)
                        print('Maxline',maxLine)
                        if ((0.1*width<distant<0.7*width)and(0.5<lengthOfl1/maxLine<=1)and(0.7<lengthOfl2/maxLine<=1)):
                            cv2.line(img_with_edges_rgb, (l1[0], l1[1]), (l1[2], l1[3]), (0,0,255))
                            cv2.line(img_with_edges_rgb, (l2[0], l2[1]), (l2[2], l2[3]), (0,0,255))
                            pts = np.array([[l1[0],l1[1]],[l1[2],l1[3]],[l2[2],l2[3]],[l2[0],l2[1]]], np.int32)
                            cv2.polylines(img,[pts],True,(0,255,0),4)
                            cv2.imwrite("./Goalpost_detected/frame%d.jpg"%count,img)
                            break
                        else:
                            continue
                break
    
    cv2.imwrite("./Frames/frame%d.jpg" % count, img)
    print ('Read a new frame: ', success)     # save frame as JPEG file	
    count += 1
    cv2.imshow('Match Detection',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success,img = vidcap.read()

# cv2.imshow("Orginal",img)
# cv2.imshow("mask",mask)s
# cv2.imshow("masked",maskedImg)
# cv2.imshow("CannyEdged",img_with_edges)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
vidcap.release()
cv2.destroyAllWindows()