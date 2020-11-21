import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plots(history):
  '''
  Plots a 4X4 plot of loss and accuracy for age and gender

  Params:
  history: Model history for which the plots are to be made

  Returns: None
  '''

  print('\nPlots\n')

  plt.subplot(221)
  plt.title('Age Loss')
  plt.plot(history.history['age_output_loss'],color= 'blue', label= 'train')
  plt.plot(history.history['val_age_output_loss'], color= 'red', label= 'test')
  plt.legend()

  plt.subplot(222)
  plt.title('Gender Loss')
  plt.plot(history.history['gender_output_loss'],color= 'blue', label= 'train')
  plt.plot(history.history['val_gender_output_loss'], color= 'red', label= 'test')
  plt.legend()

  plt.subplot(223)
  plt.title('Age Accuracy')
  plt.plot(history.history['age_output_accuracy'],color= 'blue', label= 'train')
  plt.plot(history.history['val_age_output_accuracy'], color= 'red', label= 'test')
  plt.xlabel('# of Epochs')
  plt.legend()

  plt.subplot(224)
  plt.title('Gender Accuracy')
  plt.plot(history.history['gender_output_accuracy'],color= 'blue', label= 'train')
  plt.plot(history.history['val_gender_output_accuracy'], color= 'red', label= 'test')
  plt.xlabel('# of Epochs')
  plt.legend()

  plt.tight_layout()
  plt.show()