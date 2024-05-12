''''

#### Generative Adversarial Networks (GAN) is modified for the purpose of classification of Fire and Non-Fire in Space capsule settings.
The project was a part of a ISRO Project at PES University, Banaglore, India.

Paper: https://ieeexplore.ieee.org/document/9993503

'''

from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint
#import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation

from PIL import Image , ImageOps
import os
import numpy as np

from matplotlib import pyplot as plt
from keras import backend as K

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

import cv2
import time

from sklearn.metrics import confusion_matrix
import seaborn as sns




#Extracting the frames from the video stream

vidcap = cv2.VideoCapture('/content/drive/MyDrive/Summer_Internship/realtime_fire/Fire_video/output20220719-141454.avi')
success,image = vidcap.read()
count = 0
success = True
while success:
  crop  = image[1:640 , 240:480] 
  cv2.imwrite("/content/drive/MyDrive/Summer_Internship/realtime_fire/lepton/train/fire%d.jpg" % count, crop)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1
print(count)

# Convert the images into .npz format which is used to train and test the GAN model
#### Training
def load_custom():
  label_1 = []
  vectorized_images_x = []
  count_1 = 0
  count=0
  path_to_files_train = "/content/drive/MyDrive/Summer_Internship/realtime_fire/lepton/image/"
  for img in os.listdir(path_to_files_train):
    if 'non_fire' in img:            #if your image name contain 'contrast'
      label_1.append((0))
      image_1 = Image.open(path_to_files_train + img)
      image_1 = image_1.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_1)
      vectorized_images_x.append(image_array)
      count=count+1
    elif 'fire' in img:
      label_1.append((1))
      image_1 = Image.open(path_to_files_train + img)
      image_1 = image_1.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_1)
      vectorized_images_x.append(image_array)
      count_1= count_1+1
    elif 'image_' in img:            #if your image name contain 'contrast'
      label_1.append((0))
      image_1 = Image.open(path_to_files_train + img)
      image_1 = image_1.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_1)
      vectorized_images_x.append(image_array)
      count=count+1

  print(count_1+count)
  print(count)
  print(count_1)
  np.savez("/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_train_lepton_final28.npz",DataX=vectorized_images_x,trainy=label_1)

load_dadaset=load_custom()

### Testing 
def test_custom():
  label_2 = []
  vectorized_images_y = []
  count_1 = 0
  count=0
  path_to_files_train = "/content/drive/MyDrive/Summer_Internship/realtime_fire/lepton/test/"
  for img in os.listdir(path_to_files_train):
    if 'non_fire' in img:            #if your image name contain 'contrast'
      label_2.append((0))
      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count= count+1
    elif 'fire' in img:
      label_2.append((1))
      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count_1= count_1+1
  print(count_1+count)
  print(count)
  print(count_1)
  np.savez("/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_final28.npz",Datay=vectorized_images_y,testy=label_2)

test_custom()

### This oart of the code is for training and validating the model
# It will train the model and validate it every epoch/10 times and saves the models

a1=[]
a2=[]
a3=[]
a4=[]
a5=[]
a6=[]
z=[]
ganop=[]
testacc=[]
#GLOBAL VARIABLE FOR NUMBER OF CLASSES , BATCHES , EPOCHS
n_epochs = 200
n_classes = 2
n_batch = 200
latent_dim = 100
n_samples = 200

def define_generator(latent_dim):
	
  in_lat = Input(shape=(latent_dim,))
	#Start with enough dense nodes to be reshaped and ConvTransposed to 28x28x1
  n_nodes = 256 * 7 * 7
  X = Dense(n_nodes)(in_lat)
  X = LeakyReLU(alpha=0.2)(X)
  X = Reshape((7,7, 256))(X)
	
  X = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(X) #14x14x128
  X = LeakyReLU(alpha=0.2)(X)
	
  X = Conv2DTranspose(64, (3,3), strides=(1,1), padding='same')(X) #14x14x64
  X = LeakyReLU(alpha=0.2)(X)
	# output
  out_layer = Conv2DTranspose(1, (3,3), strides=(2,2), activation='tanh', 
                             padding='same')(X) #28x28x1
	# define model
  model = Model(in_lat, out_layer)
  model.summary()
  return model

  
def define_discriminator(in_shape=(28,28,1), n_classes=n_classes):
    in_image = Input(shape=in_shape)
    X = Conv2D(32, (3,3), strides=(2,2), padding='same')(in_image)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Conv2D(64, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Conv2D(128, (3,3), strides=(2,2), padding='same')(X)
    X = LeakyReLU(alpha=0.2)(X)
    print(X)
    X = Flatten()(X)
    X = Dropout(0.4)(X) #Consider adding more dropout layers to minimize overfitting - remember we work with limited labeled data. 
    X = Dense(n_classes)(X)
    print(X)
    model = Model(inputs=in_image, outputs=X)
    model.summary()
    return model
define_discriminator()

def define_sup_discriminator(disc):
    model=Sequential()
    model.add(disc)
    model.add(Activation('softmax'))
    #Let us use sparse categorical loss so we dont have to convert our Y to categorical
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5, epsilon=1e-07), 
                  loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    return model


# custom activation function for the unsupervised discriminator
#D(x) = Z(x) / (Z(x) + 1) where Z(x) = sum(exp(l(x))). l(x) is the output from sup discr. prior to softmax
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x /(Z_x+1)
    
    return D_x

def define_unsup_discriminator(disc):
    model=Sequential()
    model.add(disc)
    #model.add(Activation('softmax'))
    model.add(Lambda(custom_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-04, beta_1=0.5, epsilon = 1e-07))
    return model
# define the combined generator and discriminator model, for updating the generator
def define_gan(gen_model, disc_unsup):
	
  disc_unsup.trainable = True # make unsup. discriminator not trainable
  gan_output = disc_unsup(gen_model.output) #Gen. output is the input to disc. 
  model = Model(gen_model.input, gan_output)
  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=2e-04, beta_1=0.5, epsilon=1e-07))
  ganop.append(gan_output)
  return model

#gan_model = define_gan(gen_model, disc_unsup)
#print("GAN MODEL SUMMARY::")
#print(gan_model.summary())

# load the images
def load_real_samples(n_classes=n_classes):
  path = "/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_train_lepton_final28.npz"
  with np.load(path , allow_pickle=True) as data:
    #load DataX as train_data
    trainX = data['DataX']
    trainy = data['trainy'] 
    (trainX, trainy) = (trainX, trainy)
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5  # scale from [0,255] to [-1,1] as we will be using tanh activation. 
    print(X.shape, trainy.shape)
    return [X, trainy]


#USED TO SELECT IMAGES. BUT IN OUR CASE ITS CHOOSING ALL THE 4 IMAGES
def select_supervised_samples(dataset, n_samples=n_samples, n_classes=n_classes):
  X, y = dataset
  X_list, y_list = list(), list()
  X_list = X
  y_list = y
  #print(X_list)
  #print(y_list)
  return asarray(X_list), asarray(y_list)    


# Pick real samples from the dataset.
#Return both images and corresponding labels in addition to y=1 indicating that the images are real. 
#Remember that we will not use the labels for unsupervised, only used for supervised. 
def generate_real_samples(dataset, n_samples=n_samples):
    
	images, labels = dataset
	ix = randint(0, images.shape[0], n_samples)
	X, labels = images[ix], labels[ix] #Select random images and corresponding labels
	y = ones((n_samples, 1)) #Label all images as 1 as these are real images. (for the discriminator training) 
	return [X, labels], y

# generate latent points, to be used as inputs to the generator. 
def generate_latent_points(latent_dim, n_samples=n_samples):
  z_input = randn(latent_dim * n_samples)
  z_input = z_input.reshape(n_samples, latent_dim) # reshape for input to the network
  #print(z_input)
  #z.append(z_input)
  return z_input

# Generate fake images using the generator and above latent points as input to it. 
#We do not care about labeles so the generator will not know anything about the labels. 
def generate_fake_samples(generator, latent_dim, n_samples=n_samples):
	
  z_input = generate_latent_points(latent_dim, n_samples)
  fake_images = generator.predict(z_input)
	# create class labels
  y = zeros((n_samples, 1)) #Label all images as 0 as these are fake images. (for the discriminator training) 
  #print(fake_images)

  return fake_images, y

# report accuracy and save plots & the model periodically. 
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset, n_samples=n_samples):
	# Generate fake images
  B, _ = generate_fake_samples(gen_model, latent_dim, n_samples)
  B = (B + 1) / 2.0 # scale to [0,1] for plotting
  A, _ = select_supervised_samples(dataset, n_samples=n_samples, n_classes=n_classes)
  A, _ = load_real_samples(n_classes)
  A = (A+1)/2.0
  X, y = dataset
  X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, n_batch)
  _, acc = disc_sup.evaluate(X, y, verbose=0)
  a1.append(acc)
  print('Discriminator Accuracy: %.3f%%' % (acc * 100))
  #print('Unsup Disc Output')
  #print(ganop)
  #ac = disc_unsup.evaluate(X_fake , y_fake, verbose=0)
  #print('Unsup Discriminator Accuracy: %.3f%%' % (ac * 100))



# train the generator and discriminator
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=n_epochs, n_batch=n_batch):
	
    # select supervised dataset for training.
   
  X_sup, y_sup = select_supervised_samples(dataset)
	#print(X_sup.shape, y_sup.shape)
	
  bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# iterations
  n_steps = bat_per_epo * n_epochs
	
  #half_batch = int(n_batch / 2)
  print('n_epochs=%d, n_batch=%d, b/e=%d, steps=%d' % (n_epochs, 
                                                              n_batch, 
                                                              bat_per_epo, n_steps))
	
    #  enumerate epochs
  for j in range(n_epochs):
    for i in range(bat_per_epo):
		# update supervised discriminator (disc_sup) on real samples.
        #Remember that we use real labels to train as this is supervised. 
        #This is the discriminator we really care about at the end.
        #Also, this is a multiclass classifier, not binary. Therefore, our y values 
        #will be the real class labels for MNIST. (NOT 1 or 0 indicating real or fake.)
      [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], n_batch)
      sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)

          
		# update unsupervised discriminator (disc_unsup) - just like in our regular GAN.
        #Remember that we will not train on labels as this is unsupervised, just binary as in our regular GAN.
        #The y_real below indicates 1s telling the discriminator that these images are real. 
        #do not confuse this with class labels. 
        #We will discard this discriminator at the end. 
      [X_real, _], y_real = generate_real_samples(dataset, n_batch) 
      d_loss_real  = disc_unsup.train_on_batch(X_real, y_real)
        #Now train on fake. 
      X_fake, y_fake = generate_fake_samples(gen_model, latent_dim, n_batch)
      d_loss_fake  = disc_unsup.train_on_batch(X_fake, y_fake)
        
		# update generator (gen) - like we do in regular GAN.
        #We can discard this model at the end as our primary goal is to train a multiclass classifier (sup. disc.)
      X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
      gan_loss = gan_model.train_on_batch(X_gan, y_gan)
      a2.append(sup_acc)
      a3.append(gan_loss)
      a4.append(sup_loss)
      a5.append(d_loss_real)
      a6.append(d_loss_fake)

		# evaluate the model performance every 'epoch'
    #plot_history(a4, a5, a3, a2) 
    # record history
   
		# summarize loss on this batch
    print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (j+1, sup_loss, sup_acc*100, d_loss_real, d_loss_fake, gan_loss))
		# evaluate the model performance periodically


    if (j+1) % ((n_epochs/10) * 1) == 0:
      summarize_performance(i, gen_model, disc_sup, latent_dim, dataset)

      filename2 = 'gen_model_%04d.h5' % (j+1)
      gen_model.save(filename2)
	# save the Discriminator (classifier) model
      filename3 = 'disc_sup_%04d.h5' % (j+1)
      disc_sup.save(filename3)

      filename4 = 'disc_unsup_%04d.h5' %(j+1)
      disc_unsup.save(filename4)
      print('>Saved: %s, %s and %s' % ( filename2, filename3 , filename4))
      from keras.models import load_model


# load the model
#disc_sup_trained_model = load_model('disc_sup_1000.h5')
      disc_sup_trained_model = load_model('disc_sup_%04d.h5' % (j+1))

# load the dataset
      path = "/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_firefinal28.npz"
      with np.load(path) as data:
    #load DataX as train_data
        testX = data['Datay']
        testy = data['testy'] 
      (testX, testy) = (testX , testy) 

# expand to 3d, e.g. add channels
      testX = expand_dims(testX, axis=-1)

# convert from ints to floats
      testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
      testX = (testX - 127.5) / 127.5

# evaluate the model
      _, test_acc = disc_sup_trained_model.evaluate(testX, testy, verbose=0)
      print('Test Accuracy: %.3f%%' % (test_acc * 100))
      testacc.append(test_acc)
      y_pred_test = disc_sup_trained_model.predict(testX)
#print(y_pred_test)
      prediction_test = np.argmax(y_pred_test, axis=1)
      print(prediction_test)

latent_dim = 100
# create the discriminator models
disc=define_discriminator() #Bare discriminator model... 
disc_sup=define_sup_discriminator(disc) #Supervised discriminator model
disc_unsup=define_unsup_discriminator(disc) #Unsupervised discriminator model. 
#gen_model = load_model('disc_sup_1000.h5')
gen_model = define_generator(latent_dim) #Generator
gan_model = define_gan(gen_model, disc_unsup) #GAN
dataset = load_real_samples() #Define the dataset by loading real samples. (This will be a list of 2 numpy arrays, X and y)
	# save the generator model



def plot_history(a4, a5, a3, a2):
	# plot loss
  plt.subplot(2, 1, 1)
  plt.plot(a4, label='sup_loss')
  plt.plot(a5, label='d-real_loss')
  plt.plot(a6, label='d-fake_loss')
  plt.plot(a3, label='gen')
  plt.legend()
	# plot discriminator accuracy
  plt.subplot(2, 1, 2)
  plt.plot(a2, label='sup_acc')
  #plt.plot(testacc, label='test_acc')
	#pyplot.plot(a2_hist, label='acc-fake')
  plt.legend()
	# save plot to file
  plt.savefig('./plot_line_plot_loss.png')
  plt.show()

def testepoch(testacc):
  plt.subplot(2,1,1)
  plt.plot(testacc , label ='test accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  plt.savefig('./epochvacc.png')
  plt.show()
	#plt.close()
# train the model

%time l = train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs, n_batch)
plot_history(a4, a5, a3, a2)
testepoch(testacc)

### This will create a dataset without labels and the model will be fed this dataset to check its accuracy

def test_custom():
  label_2 = []
  vectorized_images_y = []
  count_1 = 0
  count=0
  path_to_files_train = "/content/drive/MyDrive/Summer_Internship/realtime_fire/lepton/realtime/"
  for img in os.listdir(path_to_files_train):
    if 'non_fire' in img:            #if your image name contain 'contrast'

      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count= count+1
    elif 'fire' in img:

      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count_1= count_1+1
  print(count_1+count)
  np.savez("/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_testfinal28.npz",Datay=vectorized_images_y)

test_custom()

### Similarly for unseen images with label

def test_custom1():
  label_2 = []
  vectorized_images_y = []
  count_1 = 0
  count=0
  path_to_files_train = "/content/drive/MyDrive/Summer_Internship/realtime_fire/lepton/testing/"
  for img in os.listdir(path_to_files_train):
    if 'non_fire' in img:            #if your image name contain 'contrast'
      label_2.append((0))
      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count= count+1
    elif 'fire' in img:
      label_2.append((1))
      image_2 = Image.open(path_to_files_train + img)
      image_2 = image_2.resize((28,28), Image.ANTIALIAS).convert('L')
      image_array = np.array(image_2)
      vectorized_images_y.append(image_array)
      count_1= count_1+1
  print(count_1+count)
  print(count)
  print(count_1)
  np.savez("/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_realfinal128.npz",Datay=vectorized_images_y,testy=label_2)

test_custom1()

# load the model
#disc_sup_trained_model = load_model('disc_sup_0001.h5')
disc_sup_trained_model = load_model('disc_sup_%04d.h5' % (n_epochs))
#disc_sup_trained_model = load_model('/content/drive/MyDrive/Summer_Internship/realtime_fire/models_thermal/disc_sup_0450_989.h5')
# load the dataset
path = "/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_testfinal28.npz"
#path = "/content/drive/MyDrive/Summer_Internship/mnistlikedataset_realtime_fire_test_lepton_realfinal128.npz"
with np.load(path) as data:
    #load DataX as train_data
  testX = data['Datay']
(testX) = (testX) 

# expand to 3d, e.g. add channels
testX = expand_dims(testX, axis=-1)

# convert from ints to floats
testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
testX = (testX - 127.5) / 127.5
'''
for i in range(1):
  plt.subplot(10, 10, 1 + i)
  plt.axis('off')
  plt.imshow(testX[i, :, :, 0], cmap='gray_r')
  plt.show()
'''
# evaluate the model
print(testX.shape)
# Predicting the Test set results
%time l =  disc_sup_trained_model.predict(testX)
#print(y_pred_test)
y_pred_test_1 = disc_sup_trained_model.predict(testX)
prediction_test_1 = np.argmax(y_pred_test_1, axis=1)
print(prediction_test_1)




#### Graphs 
'''1) Epochs vs Gen Loss 2) Epochs vs unsupervised discriminator loss-fake 3) Epochs vs unsupervised discriminator loss-real 4) Epochs vs supervised discriminator loss 5) Epochs vs supervised discriminator accuracy'''

c=list(range(1,n_epochs+1))
plt.plot(c,a3)
plt.xlabel('Epochs')
plt.ylabel('Gen loss')
plt.show()
plt.plot(c,a6)
plt.xlabel('Epochs')
plt.ylabel('unsupervised discriminator loss-fake')
plt.show()
plt.plot(c,a5)
plt.xlabel('Epochs')
plt.ylabel('unsupervised discriminator loss-real')
plt.show()
plt.plot(c,a4)
plt.xlabel('Epochs')
plt.ylabel('supervised discriminator loss')
plt.show()
plt.plot(c,a2)
plt.xlabel('Epochs')
plt.ylabel('supervised discriminator accuracy')
plt.show()

def plot_history(a4, a5, a3, a2):
	# plot loss
  plt.subplot(2, 1, 1)
  plt.plot(a4, label='sup_loss')
  plt.plot(a5, label='d-real_loss')
  plt.plot(a6, label='d-fake_loss')
  plt.plot(a3, label='gen')
  plt.legend()
	# plot discriminator accuracy
  plt.subplot(2, 1, 2)
  plt.plot(a2, label='sup_acc')
  plt.legend()
	# save plot to file
  plt.savefig('./plot_line_plot_loss.png')
  plt.show()

def testepoch(testacc):
  plt.subplot(2,1,1)
  plt.plot(testacc , label ='test accuracy')
  plt.xlabel('Epochs')
  plt.legend()
  plt.savefig('./epochvacc.png')
  plt.show()

plot_history(a4, a5, a3, a2)
testepoch(testacc)

## Confusion Matrix

cm = confusion_matrix(testy, prediction_test_1)
sns.heatmap(cm, annot=True)
tn,fp,fn,tp = confusion_matrix(testy, prediction_test_1).ravel()

print("True Positive Rate" , tp)
print("False Positive Rate" , fp)
print("True Negative Rate" , tn)
print("False Negative Rate" , fn)

def compute_accuracy(tp,tn,fp,fn):
  return((tn+tp)*100/float(tn+fn+tp+fp))
print("Accuracy = " , compute_accuracy(tp,tn,fp,fn))

def compute_precision(tp,fp):
  return ((tp*100)/float(tp+fp))
print("Precision = " , compute_precision(tp,fp))

def compute_recall(tp,fn):
  return ((tp*100)/float(tp+fn))
print("Recall = ", compute_recall(tp,fn))

def compute_f1_score(tp,tn,fp,fn):
  precision = compute_precision(tp,fp)/100
  recall = compute_recall(tp,fn)/100
  f1_score = (2*precision*recall)/(precision+recall)
  return f1_score

print("F1 score = ", compute_f1_score(tp,tn,fp,fn))