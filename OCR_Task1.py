import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\College\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

img = cv2.imread("testing2.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img))
img_h,img_w,_ = img.shape
boxes = pytesseract.image_to_boxes(img)
for lines in boxes.splitlines():
	char,x,(y),w,h,_ = (lines.split(" "))
	x,y,w,h = int(x),int(y),int(w),int(h)
	# print(type(x),type(y))
	cv2.rectangle(img,(x,img_h-y),(w,img_h-h),(0,255,0),1)


cv2.imshow("Output",cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
cv2.waitKey(0)