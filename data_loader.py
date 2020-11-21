import os
import pandas as pd
import cv2
import numpy as np


class DataLoader:
	def __init__(self, data_dir):
		'''
		Initializes a DataLoader object

		Args
		data_dir: Dataset file path

		Returns: None
		'''
		self.data_dir = data_dir
		self.files = os.listdir(data_dir)
		self.images = []
		self.age_labels = []
		self.gender_labels = []

	def load(self):
		'''
		Loads the images and labels from the initialized data directory

		Args: None
		Returns:
		images: all the images as a numpy array
		age_labels: age groups of all images as a numpy array
		gender_labels: gender of all the images as a numpy array
		'''
		self.images = []
		self.age_labels = []
		self.gender_labels = []

		for f in self.files:

			#Read and resize image to (64, 64).
			img = cv2.imread(self.data_dir+'/'+f,0)
			img = cv2.resize(img,(64,64))
			img = img.reshape((img.shape[0], img.shape[1], 1))
			self.images.append(img)

			age = int(f.split('_')[0])
			gender = int(f.split('_')[1]) #1 - Female 0 - Male.

			if gender == 1:
				self.gender_labels.append('Female')
			else:
				self.gender_labels.append('Male')

			if age <= 14: 
				self.age_labels.append('0 - 14')
			elif age > 14 and age <= 25:
				self.age_labels.append('14 - 25')
			elif age > 25 and age <= 40:
				self.age_labels.append('25 - 40')
			elif age > 40 and age <= 60:
				self.age_labels.append('40 - 60')
			else:
				self.age_labels.append('60+')

		self.images = np.asarray(self.images)
		self.age_labels = np.array(self.age_labels)
		self.gender_labels = np.array(self.gender_labels)

		return self.images, self.age_labels, self.gender_labels