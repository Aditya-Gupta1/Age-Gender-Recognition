import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from define_model import AgeGenderModel
from data_loader import DataLoader
import utils

class AgeGenderEstimation:
	def __init__(self, data_dir, input_shape, optimizer= 'adam', 
				 metrics= ['accuracy'], batch_size= 64, epochs= 50, 
				 verbose= 1, validation_split= None, save= True, 
				 save_dir = 'models/age_gender_model'):

		self.data_dir = data_dir
		self.input_shape = input_shape
		self.optimizer= optimizer
		self.metrics= metrics
		self.batch_size = batch_size
		self.epochs = epochs
		self.verbose = verbose
		self.validation_split = validation_split
		self.save = save
		self.save_dir = save_dir
		self.images = None
		self.age_labels = None
		self.gender_labels = None
		self.model = None
		self.model_history = None
		self.age_binarizer = LabelBinarizer()
		self.gender_binarizer = LabelBinarizer()
		self.losses = {'age_output': 'categorical_crossentropy',
		'gender_output': 'binary_crossentropy'
		}

		self.loader = DataLoader(self.data_dir)
		self.modeler = AgeGenderModel(input_shape= self.input_shape)

		print("Program Instantiated!")

	def get_data(self):

		self.images, self.age_labels, self.gender_labels = self.loader.load()

		self.age_labels = self.age_binarizer.fit_transform(self.age_labels)
		self.gender_labels = self.gender_binarizer.fit_transform(self.gender_labels)

		print("Data Loaded Successfully!")


	def get_model(self):

		self.model = self.modeler.get_model()

		self.model.compile(loss = self.losses, optimizer= self.optimizer, metrics= self.metrics)

		print("Model Loaded and Compiled Succesfully!")

	def train(self):

		self.model_history = self.model.fit(self.images, 
					  						{'age_output': self.age_labels, 'gender_output': self.gender_labels},
					  						epochs = self.epochs,
					  						validation_split = self.validation_split, 
					  						batch_size= self.batch_size,
					  						verbose = self.verbose)

		if self.save and not self.save_dir:
			print("Path to save the model not found!")
			return

		if self.save and self.save_dir:
			self.model.save(self.save_dir)
			print("Model Save Successfully!")

	def plot_results(self):

		utils.plots(self.model_history)