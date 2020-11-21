from training import AgeGenderEstimation
import argparse
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

ap = argparse.ArgumentParser()
ap.add_argument('-dr', '--datadir', required= True, help= 'path to input folder')
ap.add_argument('-is', '--inputshape', required= True, help= 'input shape of model')
ap.add_argument('-op', '--optimizer', required= False, help= 'model optimizer', default= 'adam')
ap.add_argument('-m', '--metrics', required= False, help= 'model metrics', default= ['accuracy'])
ap.add_argument('-bs', '--batchsize', required= False, help= 'model batch size', default= 64)
ap.add_argument('-e', '--epochs', required= False, help= 'number of epochs for training', default= 50)
ap.add_argument('-v', '--verbosity', required= False, help= 'model verbosity level', default= 1)
ap.add_argument('-vs', '--validationsplit', required= False, help= 'validation split', default= None)
ap.add_argument('-s', '--save', required= False, help= 'whether to save the model', default= True)
ap.add_argument('-sd', '--savedir', required= False, help= 'path to save the model', default= 'models/age_gender_model')

args= vars(ap.parse_args())

shape = args['inputshape']
shape = list(map(int, shape.strip().split(',')))
args['validationsplit'] = float(args['validationsplit'])

project = AgeGenderEstimation(data_dir= args['datadir'],
							  input_shape= shape,
							  optimizer= args['optimizer'],
							  metrics= args['metrics'],
							  batch_size= args['batchsize'],
							  epochs= args['epochs'],
							  verbose= args['verbosity'],
							  validation_split= args['validationsplit'],
							  save= args['save'],
							  save_dir= args['savedir'])

project.get_data()
project.get_model()
project.train()