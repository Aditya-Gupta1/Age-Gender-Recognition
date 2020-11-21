from evaluate import Evaluate
import argparse
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required= True, help= 'path to model')
ap.add_argument('-i', '--image', required= True, help= 'path to input image')

args = vars(ap.parse_args())

test = Evaluate(args['image'], args['model'])
test.load_image()
found_faces = test.extract_faces()

for idx, face in enumerate(found_faces):
	
	age, gender, age_prob, gender_prob = test.predict(face)
	print("Face: ", idx+1)
	print(age, gender, age_prob, gender_prob)
