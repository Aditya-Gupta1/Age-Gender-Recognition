import tensorflow
from tensorflow.keras.models import load_model
import cv2
from mtcnn import MTCNN
import numpy as np

class Evaluate:
	def __init__(self, image_path= None, image= None, model_path=None, model= None):
		'''
		Instantiates an object of a class to perform evaluation.

		Params: None
		Returns: None
		'''
		self.image_path = image_path
		self.image = image
		self.face_detector = MTCNN()
		self.age_labels = ['0 - 14','14 - 25','25 - 40','40 - 60','60+']
		self.gender_labels = ['Female', 'Male']
		self.model_path = model_path

		if self.model_path:
			self.model = load_model(self.model_path)
		else:
			self.model = model

	def load_image(self):
		'''
		Loads an image from given path

		Params: None
		Returns: None
		'''
		if self.image_path:

			self.image = cv2.imread(self.image_path)
			self.model_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

	def extract_faces(self, face_threshold= 0.9):
		'''
		Extracts faces from a given image using mtcnn

		Params:None

		Returns:
		faces: list of bounding boxes of faces having confidence more than 90%
		'''
		self.face_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
		self.detected_faces = self.face_detector.detect_faces(self.face_image)
		self.faces = []

		for item in self.detected_faces:

			if item['confidence'] >= face_threshold:
				self.faces.append(item['box'])

		return self.faces

	def predict(self, box):
		'''
		Make predictions on the image bounded by a box

		Params:
		box: list of x,y,w,h

		Returns:
		age: age of the face in a given box
		gender: gender of the face in a given box
		age_prob: probability of a predicted age
		gender_prob: probability of predicted gender
		'''	
		x, y, w, h = box
		image = self.model_image[y:y+h, x:x+w]
		image = cv2.resize(image, (64, 64))
		image = np.array(image)
		image = np.reshape(image, (1, 64, 64, 1))

		age_pred, gender_pred = self.model.predict(image)

		age_pred = np.squeeze(age_pred)
		gender_pred = np.squeeze(gender_pred)
		gender_pred = np.array([1-gender_pred, gender_pred])

		age = self.age_labels[np.argmax(age_pred)]
		age_prob = age_pred[np.argmax(age_pred)]
		gender = self.gender_labels[np.argmax(gender_pred)]
		gender_prob = gender_pred[np.argmax(gender_pred)]

		return age, gender, age_prob, gender_prob