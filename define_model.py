import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Conv1D


class AgeGenderModel:

	def __init__(self, input_shape):
		'''
		Initializes an instance of the class to create a model

		Params:
		input_shape: input dimensions of the model

		Returns: None
		'''
		self.input_shape = input_shape

	def get_age_branch(self, x):
		'''
		Generates the model for Age Estimation

		Params:
		x: previous input layer

		Returns: None
		'''

		age_output = Conv2D(128, (3,3), activation= 'relu')(x)
		age_output = MaxPool2D(pool_size= (3,3))(age_output)
		age_output = Dropout(0.3)(age_output)

		age_output = Flatten()(age_output)
		age_output = Dropout(0.4)(age_output)

		age_output = Dense(64, activation= 'relu')(age_output)
		age_output = Dropout(0.3)(age_output)

		age_output = Dense(256, activation= 'relu')(age_output)
		age_output = Dropout(0.5)(age_output)

		age_output = Dense(5,activation= 'softmax', name= 'age_output')(age_output)

		return age_output

	def get_gender_branch(self, x):
		'''
		Generates the model for Age Estimation

		Params:
		x: previous input layer

		Returns: None
		'''
		gender_output = Flatten()(x)
		gender_output = Dense(512, activation= 'relu')(gender_output)
		gender_output = Dropout(0.3)(gender_output)

		gender_output = Dense(512, activation= 'relu')(gender_output)
		gender_output = Dense(1,activation= 'sigmoid', name= 'gender_output')(gender_output)

		return gender_output

	def get_model(self):
		'''
		Defines a model

		Params: None

		Returns:
		model: a CNN model
		'''

		input = Input(shape= self.input_shape)

		x = Conv2D(32, (3,3), activation= 'relu')(input)
		x = Conv2D(64, (3,3), activation= 'relu', )(x)

		x = MaxPool2D(pool_size= (3,3))(x)
		x = Dropout(0.3)(x)

		age_output = self.get_age_branch(x)

		gender_output = self.get_gender_branch(x)

		model = Model(inputs= input, outputs= [age_output, gender_output])

		return model