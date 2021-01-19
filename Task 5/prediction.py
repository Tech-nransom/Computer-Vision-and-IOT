import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # CONVERT TO GRAYSCALE
    img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img


def Result(classNo):
	if   classNo == 0: return 'Speed Limit 20 km/h'
	elif classNo == 1: return 'Speed Limit 30 km/h'
	elif classNo == 2: return 'Speed Limit 50 km/h'
	elif classNo == 3: return 'Speed Limit 60 km/h'
	elif classNo == 4: return 'Speed Limit 70 km/h'
	elif classNo == 5: return 'Speed Limit 80 km/h'
	elif classNo == 6: return 'End of Speed Limit 80 km/h'
	elif classNo == 7: return 'Speed Limit 100 km/h'
	elif classNo == 8: return 'Speed Limit 120 km/h'
	elif classNo == 9: return 'No passing'
	elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
	elif classNo == 11: return 'Right-of-way at the next intersection'
	elif classNo == 12: return 'Priority road'
	elif classNo == 13: return 'Yield'
	elif classNo == 14: return 'Stop'
	elif classNo == 15: return 'No vechiles'
	elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
	elif classNo == 17: return 'No entry'
	elif classNo == 18: return 'General caution'
	elif classNo == 19: return 'Dangerous curve to the left'
	elif classNo == 20: return 'Dangerous curve to the right'
	elif classNo == 21: return 'Double curve'
	elif classNo == 22: return 'Bumpy road'
	elif classNo == 23: return 'Slippery road'
	elif classNo == 24: return 'Road narrows on the right'
	elif classNo == 25: return 'Road work'
	elif classNo == 26: return 'Traffic signals'
	elif classNo == 27: return 'Pedestrians'
	elif classNo == 28: return 'Children crossing'
	elif classNo == 29: return 'Bicycles crossing'
	elif classNo == 30: return 'Beware of ice/snow'
	elif classNo == 31: return 'Wild animals crossing'
	elif classNo == 32: return 'End of all speed and passing limits'
	elif classNo == 33: return 'Turn right ahead'
	elif classNo == 34: return 'Turn left ahead'
	elif classNo == 35: return 'Ahead only'
	elif classNo == 36: return 'Go straight or right'
	elif classNo == 37: return 'Go straight or left'
	elif classNo == 38: return 'Keep right'
	elif classNo == 39: return 'Keep left'
	elif classNo == 40: return 'Roundabout mandatory'
	elif classNo == 41: return 'End of no passing'
	elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'



model = tf.keras.models.load_model("final_model.h5")
imgOrignal = cv2.imread("testing/image5.jpg")
font = cv2.FONT_HERSHEY_COMPLEX
img = np.asarray(imgOrignal)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img)
img = img.reshape(1, 32, 32, 1)
predictions = model.predict(img)
probabilityValue =np.amax(predictions)
print("\tClass-->",Result(np.argmax(predictions)))
print("\tProbability--->",round(probabilityValue*100,2),"%")
imgOrignal = cv2.cvtColor(imgOrignal,cv2.COLOR_BGR2RGB)
plt.imshow(imgOrignal)
plt.show()