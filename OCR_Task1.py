import pytesseract
import cv2
import numpy as np
import utlis

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\College\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

img = cv2.imread("testing4.jpg",flags=0)
img_h,img_w = img.shape
# img = cv2.resize(img, (512, 512))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.threshold(img, 0, 255,cv2.THRESH_TOZERO)[1]
imgBlank = np.zeros((img_h,img_w, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
thres=utlis.valTrackbars() # GET TRACK BAR VALUES FOR THRESHOLDS
imgThreshold = cv2.Canny(imgBlur,200,10)#thres[0],thres[1]) # APPLY CANNY BLUR
cv2.imshow("ImgThresh",imgThreshold)
cv2.waitKey(0)
kernel = np.ones((5, 5))
imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # APPLY DILATION
imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
## FIND ALL COUNTOURS
imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # DRAW ALL DETECTED CONTOURS
cv2.imshow("Countour",imgContours)
cv2.waitKey(0)
# FIND THE BIGGEST COUNTOUR
biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
if biggest.size != 0:
    biggest=utlis.reorder(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[img_w, 0], [0, img_h],[img_w, img_h]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (img_w, img_h))

    #REMOVE 20 PIXELS FORM EACH SIDE
    imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
    imgWarpColored = cv2.resize(imgWarpColored,(img_w,img_h))

    # APPLY ADAPTIVE THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
    # imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
    imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
    imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)


# print(pytesseract.image_to_string(imgWarpGray))
try:
	wordBoxes = pytesseract.image_to_data(imgWarpColored)
	for x,b in enumerate(wordBoxes.splitlines()):
		if x != 0:
			b = b.split()
			# print(b)
			if len(b) == 12:
				x,y,w,h,word = int(b[6]),int(b[7]),int(b[8]),int(b[9]),b[-1]
				cv2.rectangle(imgWarpColored,(x,y),(w+x,h+y),(0,255,0),3)
				cv2.putText(imgWarpColored,word,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

	# blurred=cv2.GaussianBlur(gray,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
	# cv2.imshow("Blur",blurred)

	# edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
	# cv2.imshow("Canny",edged)




	# contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
	# contours=sorted(contours,key=cv2.contourArea,reverse=True)

	# #the loop extracts the boundary contours of the page
	# for c in contours:
	#     p=cv2.arcLength(c,True)
	#     approx=cv2.approxPolyDP(c,0.02*p,True)

	#     if len(approx)==4:
	#         target=approx
	#         break
	# approx=mapp(target) #find endpoints of the sheet

	# pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

	# op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
	# dst=cv2.warpPerspective(img,op,(800,800))


	# boxes = pytesseract.image_to_boxes(imgWarpGray)
	# for lines in boxes.splitlines():
	# 	char,x,(y),w,h,_ = (lines.split(" "))
	# 	x,y,w,h = int(x),int(y),int(w),int(h)
	# 	# print(type(x),type(y))
	# 	cv2.rectangle(imgWarpGray,(x,img_h-y),(w,img_h-h),(0,255,0),1)
	# 	cv2.putText(imgWarpGray,char,(x-15,img_h-y-15),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)


	cv2.imshow("Output",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
	cv2.imshow("Gray",imgWarpColored)
	# cv2.imshow("Final",dst)
	cv2.waitKey(0)
except:
	wordBoxes = pytesseract.image_to_data(img)
	for x,b in enumerate(wordBoxes.splitlines()):
		if x != 0:
			b = b.split()
			# print(b)
			if len(b) == 12:
				x,y,w,h,word = int(b[6]),int(b[7]),int(b[8]),int(b[9]),b[-1]
				cv2.rectangle(img,(x,y),(w+x,h+y),(0,255,0),3)
				cv2.putText(img,word,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)
	cv2.imshow("Output",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
	cv2.waitKey(0)