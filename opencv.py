
import cv2
import numpy as np
import math


hand_cascade = cv2.CascadeClassifier('Hand_haar_cascade.xml')

video = cv2.VideoCapture(0)

while True:
	check, img = video.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	blur = cv2.GaussianBlur(gray,(5,5),0)
    
	#change the threshold value according to the surrounding light condition
	retval,thresh1 = cv2.threshold(blur,80,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 

	hand = hand_cascade.detectMultiScale(thresh1,scaleFactor= 1.05, minNeighbors= 5) 
	mask = np.zeros(thresh1.shape, dtype = "uint8") 

	
	for (x,y,w,h) in hand: 
		cv2.rectangle(img,(x,y),(x+w,y+h), (0,122,122), 2) 
		cv2.rectangle(mask, (x,y),(x+w,y+h),255,-1)

	img2 = cv2.bitwise_and(thresh1, mask)
	final = cv2.GaussianBlur(img2,(7,7),0)	
	contours, hierarchy = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	cv2.drawContours(img, contours, 0, (255,255,0), 3)
	cv2.drawContours(final, contours, 0, (255,255,0), 3)

	if len(contours) > 0:
		cnt=contours[0]
		hull = cv2.convexHull(cnt, returnPoints=False)
		
		defects = cv2.convexityDefects(cnt, hull)
		count_defects = 0
		
		
		if defects is not None:
			for i in range(defects.shape[0]):
				s,e,f,a = defects[i,0]
				start = tuple(cnt[s][0])
				end = tuple(cnt[e][0])
				far = tuple(cnt[f][0])
				
				#finding length of each line when two end points are there
				a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
				b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
				c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
				
				angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57.29
				
				if angle <= 90:
				    count_defects = count_defects + 1
		
		if  count_defects == 0:
			cv2.putText(img,"1", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)
		elif count_defects == 1:
			cv2.putText(img,"2", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)
		elif count_defects == 2:
			cv2.putText(img, "3", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)
		elif count_defects == 3:
			cv2.putText(img,"4", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)
		elif count_defects == 4:
			cv2.putText(img,"5", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)
		#else:
		#	cv2.putText(img,"closed", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 0),2)

	
	cv2.imshow('original',img)
	cv2.imshow('masked_image',img2)

	k = cv2.waitKey(30) & 0xff
	#press escape to exit
	if k == 27:
		break

video.release()
cv2.destroyAllWindows()