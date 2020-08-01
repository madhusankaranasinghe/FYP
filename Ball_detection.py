#Import libraries
import cv2
import os
import numpy as np

#Reading the video
vidcap = cv2.VideoCapture('video3.mp4')
success,image = vidcap.read()

count = 0
success = True
idx = 0


#Read the video frame by frame
while success:
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imwrite("./frames/frame%d.jpg" % count, image)
    #green range
    lower_green = np.array([40,40, 40])
    upper_green = np.array([70, 255, 255])
    lower_white = np.array([0,0,200])
    upper_white = np.array([0,250,255])
    #Define a mask ranging from lower to uppper
    mask = cv2.inRange(hsv, lower_green, upper_green)
    #Do masking
    res = cv2.bitwise_and(image, image, mask=mask)
    resToGray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
	#convert to hsv to gray
    res_bgr = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    res_gray = cv2.cvtColor(res_bgr,cv2.COLOR_BGR2GRAY)
    
    #Eliminate lines
    # edges = cv2.Canny(res_gray,50,150,apertureSize = 3)
    # minLineLength = 100
    # maxLineGap = 10
    # lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    #     for x1,y1,x2,y2 in lines[0]:
    #         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    #Defining a kernel to do morphological operation in threshold image to 
    #get better output.
    kernel = np.ones([13,13],np.uint8)
    thresh = cv2.threshold(res_gray,240,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	

    #find contours in threshold image     
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        

    prev = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
	
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        approx = cv2.approxPolyDP(c,0.01*cv2.arcLength(c,True),True)
        area = cv2.contourArea(c)
        if((h>=1 and w>=1) and (h<=30 and w<=30) and ((len(approx) > 8) & (area > 30))):
            player_img = image[y:y+h,x:x+w]
            #cv2.imwrite("./playerImg/frame%d.jpg"%count,player_img)
            
            player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                #white ball  detection
            mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
            res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
            res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
            nzCount = cv2.countNonZero(res1)
        
            if(nzCount >= 1):
                # detect football
                cv2.putText(image, 'football', (x-2, y-2), font, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.imwrite("./ball_detect_frame/frame%d.jpg" % count, image)


    cv2.imwrite("./Cropped/frame%d.jpg" % count, resToGray)
    print ('Read a new frame: ', success)     # save frame as JPEG file	
    count += 1
    cv2.imshow('Match Detection',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success,image = vidcap.read()
    
vidcap.release()
cv2.destroyAllWindows()

