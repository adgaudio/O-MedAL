# Import functions
from __future__ import print_function
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input,Conv2D,GlobalAveragePooling2D,Dense,Flatten,Lambda
from keras.models import load_model,save_model,Model
from scipy.spatial.distance import cdist
from classifier5 import Classifier  
from skimage.feature import corner_peaks,corner_harris,BRIEF,ORB
from skimage.color import rgb2gray
from skimage.filters import gaussian
from sklearn.utils import shuffle
import tensorflow as tf 
from keras import backend as K
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#####################################################################################################################
# Note: Overall  dataset consists of 400 images (100 per class among four classes). 
# We are using 320 images (80 per class) as Train set+oracle set and 80 images (20 per class) as Test set

# Control Parameters

Method_type = 3                                 # 0:Baseline accuracy computation (Entire dataset) 
                                                # 1: Random sampling  2:Max Entropy measure  3:Our method
Initial_dataset_size = 200                       # Initial dataset size to sample using ORB descriptors
No_of_images_to_sample = 20                      # No of images to sample each AL iteration, common for all method types
No_of_images_entropy_method3 = 50               # No of images to sample initially from entropy measure (before using learnt feature vectors) for our method  
Distance_metric = 'euclidean'                   # Distance metric to compare model feature vectors
Create_initial_dataset = False                  # True:  Performs ORB descriptor technique (unless Load_descriptors = True) to create initial training and oracle sets from overall available dataset
                                                #        Use this to start the active learning process from stratch for a new method_type     
                                                # False: Loads most recent training and oracle sets stored in same directory
											    #        and starts from the last Active learning iteration performed for a particular method_type  
Accuracy_array_name = 'testacc_Method1.npy'     # Numpy array storage name. This records the best test accuracy acheived during each AL iteration.
                                                # Change name for different method_type
AUC_array_name = 'AUC_Method1.npy'              # Stores the AUC score on test set for each AL iteration.Change name for different method_type
Load_descriptors = True                        # Loads precomputed descriptors for all dataset images.It is useful only when Create_initial_dataset==True  

# Network Training parameters
Epochs = 5                                # no of epochs to train model for each AL iteration
AL_iterations = 32                         # no of times to implement Active learning algorithm   
Batch_size = 8
train_from_stratch = False                 # Trains model from stratch for each AL iteration
learning_rate = 1e-4                    # Learning rate for deep CNN model
Img_resol = 512                           # Input image resolution (Height == Width assumed) 
Availble_training_data_size = 765
Availble_test_data_size = 188
No_classes = 2

####################################################################################################################

# Function definitions
def add_file_to_train(Indices):
	global X_train,Y_train,X_oracle,Y_oracle,No_of_images_to_sample,Method_type,Img_resol,No_classes

	if Method_type==3:
		if K.image_data_format() == 'channels_first':
			Addition = np.reshape(np.take(X_oracle,Indices,axis=0),(1,3,Img_resol,Img_resol))
		else:
			Addition = np.reshape(np.take(X_oracle,Indices,axis=0),(1,Img_resol,Img_resol,3))
		if No_classes==2:	
			Addition1 = np.reshape(np.take(Y_oracle,Indices,axis=0),(1,))
		else:
			Addition1 = np.reshape(np.take(Y_oracle,Indices,axis=0),(1,No_classes))				
	else:	
		if K.image_data_format() == 'channels_first':
			Addition = np.reshape(np.take(X_oracle,Indices,axis=0),(No_of_images_to_sample,3,Img_resol,Img_resol))
		else:
			Addition = np.reshape(np.take(X_oracle,Indices,axis=0),(No_of_images_to_sample,Img_resol,Img_resol,3))
		if No_classes==2:	
			Addition1 = np.reshape(np.take(Y_oracle,Indices,axis=0),(No_of_images_to_sample,))
		else:
			Addition1 = np.reshape(np.take(Y_oracle,Indices,axis=0),(No_of_images_to_sample,No_classes))					

	X_train = np.concatenate((X_train,Addition),axis=0)
	Y_train = np.concatenate((Y_train,Addition1),axis=0)

	X_oracle = np.delete(X_oracle,Indices,axis=0)
	Y_oracle = np.delete(Y_oracle,Indices,axis=0)

	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_oracle.npy',X_oracle)
	np.save('Y_oracle.npy',Y_oracle)

######################################################################################################################3

def calculate_descriptors(X):
	descriptor_extractor = ORB(n_keypoints=500)
	Descriptors = []
	for i in range(len(X)):
		print ('Calculating ORB descriptor for image:{}'.format(i))
		Im = np.asarray(X[i,:,:,:],dtype='float32')
		Max = np.amax(Im)
		Im = Im/Max
		Im = rgb2gray(Im)
	
		descriptor_extractor.detect_and_extract(Im)
		Temp = descriptor_extractor.descriptors
		Descriptors.append(np.asarray(np.round(np.average(Temp,axis=0)),dtype='int32'))

	Descriptors_matrix = np.zeros([len(X),256])
	for i in range(len(X)):
		Descriptors_matrix[i,:] = Descriptors[i] 
     
	return Descriptors_matrix
##############################################################################################################

def calculate_distance(X_oracle_descriptor,X_train_descriptor):
	Distances = np.zeros([len(X_oracle_descriptor),len(X_train_descriptor)])
	for i in range(len(X_oracle_descriptor)):
		Oracle = np.reshape(X_oracle_descriptor[i,:],(1,256))
		for j in range(len(X_train_descriptor)):
			Train = np.reshape(X_train_descriptor[j,:],(1,256))
			Distances[i,j] = cdist(Train,Oracle,'euclidean')

	Distances = np.reshape(np.average(Distances,axis=1),(len(X_oracle_descriptor),))
	Sorted_distances = np.flip(np.sort(Distances,axis=0),axis=0)
	Sorted_indexes = np.flip(np.argsort(Distances,axis=0),axis=0)
	return Sorted_distances,Sorted_indexes

#############################################################################################################

def create_trainset():
	global Initial_dataset_size,X_oracle,X_oracle_descriptor,Y_oracle,Img_resol,No_classes
    
	Initial_sample = np.random.randint(0,len(X_oracle),1)

	X_train_descriptor =  np.reshape(X_oracle_descriptor[Initial_sample,:],(1,256))
	X_train = np.reshape(X_oracle[Initial_sample,:,:,:],(1,Img_resol,Img_resol,3))
	if No_classes==2:
		Y_train = np.reshape(Y_oracle[Initial_sample],(1,))
	else:
		Y_train = np.reshape(Y_oracle[Initial_sample,:],(1,No_classes))			

	X_oracle = np.delete(X_oracle,Initial_sample,axis=0)
	X_oracle_descriptor = np.delete(X_oracle_descriptor,Initial_sample,axis=0)
	Y_oracle = np.delete(Y_oracle,Initial_sample,axis=0)

	print ('Creating training dataset:')
	for i in range(Initial_dataset_size-1):
		Dist,Index = calculate_distance(X_oracle_descriptor, X_train_descriptor)

		Selected_Index = Index[0]
        
		Addition = np.reshape(X_oracle[Selected_Index,:,:,:],(1,Img_resol,Img_resol,3))
		X_train = np.concatenate((X_train,Addition),axis=0)
	
		Addition = np.reshape(X_oracle_descriptor[Selected_Index,:],(1,256))
		X_train_descriptor = np.concatenate((X_train_descriptor,Addition),axis=0)

		X_oracle = np.delete(X_oracle,Selected_Index,axis=0)
		X_oracle_descriptor = np.delete(X_oracle_descriptor,Selected_Index,axis=0)

		if No_classes==2:
			Addition = np.reshape(Y_oracle[Selected_Index],(1,))
		else:
			Addition = np.reshape(Y_oracle[Selected_Index],(1,No_classes))
							
		Y_train = np.concatenate((Y_train,Addition),axis=0)
		Y_oracle = np.delete(Y_oracle,Selected_Index,axis=0)
   
		X_train,Y_train = shuffle(X_train,Y_train,random_state=62)

	return X_train,Y_train
##############################################################################################################

def compute_feature_vectors(Indices):
	global DL_model,X_oracle,X_train
    
	Selection = np.take(X_oracle,Indices,axis=0)
	Output = GlobalAveragePooling2D()(DL_model.layers[164].output)
	Dummy_model = Model(DL_model.input,Output)

	sess = K.get_session()

	oracle_feature_vectors = np.array(sess.run(Dummy_model.output,feed_dict={Dummy_model.input:Selection}),dtype=np.float32)
	train1 = np.array(sess.run(Dummy_model.output,feed_dict={Dummy_model.input:X_train[:int(len(X_train)/3),:,:,:]}),dtype=np.float32)
	train2 = np.array(sess.run(Dummy_model.output,feed_dict={Dummy_model.input:X_train[int(len(X_train)/3):int(2*len(X_train)/3),:,:,:]}),dtype=np.float32)
	train3 = np.array(sess.run(Dummy_model.output,feed_dict={Dummy_model.input:X_train[int(2*len(X_train)/3)::,:,:]}),dtype=np.float32)
 
	train_feature_vectors = np.concatenate((train1,train2,train3),axis=0)

	return oracle_feature_vectors,train_feature_vectors
#############################################################################################################

def compare_feature_vectors(oracle_feature_vectors,train_feature_vectors,Indices):
	global No_of_images_to_sample,Distance_metric

	for i in range(No_of_images_to_sample):
		compared_distances = np.reshape(np.min(cdist(oracle_feature_vectors,train_feature_vectors,Distance_metric),axis=1),(len(oracle_feature_vectors),))  
		selected_distance_index = Indices[np.argmax(compared_distances)]
 
		Addition = np.reshape(oracle_feature_vectors[np.argmax(compared_distances),:],(1,oracle_feature_vectors.shape[1]))
		train_feature_vectors = np.concatenate((train_feature_vectors,Addition),axis=0)
		oracle_feature_vectors = np.delete(oracle_feature_vectors,np.argmax(compared_distances),axis=0)
		Indices = np.delete(Indices,np.argmax(compared_distances),axis=0)
		for j,r in enumerate(Indices):
			 if r>selected_distance_index:
				 Indices[j] = r-1
		add_file_to_train(selected_distance_index.tolist())

##############################################################################################################
def random_sample():
	global X_train,X_oracle,Y_train,Y_oracle,No_of_images_to_sample,Img_resol,No_classes

	Indice = np.arange(0,len(X_oracle),1)

	Positive_index = []
	for i in range(No_of_images_to_sample):
		I = np.random.randint(0,len(Indice),1)
		Positive_index.append(Indice[I])
		Indice = np.delete(Indice,I)

	if K.image_data_format() == 'channels_first':
		Addition = np.reshape(np.take(X_oracle,Positive_index,axis=0),(No_of_images_to_sample,3,Img_resol,Img_resol))
	else:
		Addition = np.reshape(np.take(X_oracle,Positive_index,axis=0),(No_of_images_to_sample,Img_resol,Img_resol,3)) 
	X_train = np.concatenate((X_train,Addition),axis=0)
	X_oracle = np.delete(X_oracle,Positive_index,axis=0)

	if No_classes==2:
		Addition = np.squeeze(np.take(Y_oracle,Positive_index,axis=0),axis=1)
	else:
		Addition = np.take(Y_oracle,Positive_index,axis=0)

	Y_train = np.concatenate((Y_train,Addition),axis=0)
	Y_oracle = np.delete(Y_oracle,Positive_index,axis=0)    
	np.save('X_train.npy',X_train)
	np.save('Y_train.npy',Y_train)
	np.save('X_oracle.npy',X_oracle)
	np.save('Y_oracle.npy',Y_oracle)
   
##############################################################################################################################

def compute_entropy(oracle_preds):
	global No_classes

	Entropy_measure = np.zeros([oracle_preds.shape[0],])
	if No_classes==2:
		for i,r in enumerate(oracle_preds):
			Entropy_measure[i] = -r*np.log2(r) + (1-r)*np.log2(1-r)
	else:		
		for i in range(len(oracle_preds)):
			for j in oracle_preds[i,:]:
				if j!=0.0 and j!=1.0:
					Entropy_measure[i] += -j*np.log2(j)

	return Entropy_measure

###############################################################################################################################

def Active_learning():
	global X_oracle,X_train,Y_oracle,Y_train,Method_type,DL_model,No_of_images_to_sample,No_of_images_entropy_method3

	if Method_type==1:
		random_sample()

	elif Method_type==2:
		oracle_preds = DL_model.predict(X_oracle,batch_size=32,verbose=1)
		Entropy = compute_entropy(oracle_preds)

		Lowest_entropy_indices = np.flip(np.argsort(Entropy),axis=0)[:No_of_images_to_sample]
		add_file_to_train(Lowest_entropy_indices)

	elif Method_type==3:
		oracle_preds = DL_model.predict(X_oracle,batch_size=32,verbose=1)
		Entropy = compute_entropy(oracle_preds)
		Lowest_entropy_indices = np.flip(np.argsort(Entropy),axis=0)[:No_of_images_entropy_method3]
        
		oracle_feature_vectors,train_feature_vectors = compute_feature_vectors(Lowest_entropy_indices)
		compare_feature_vectors(oracle_feature_vectors,train_feature_vectors,Lowest_entropy_indices)

##################################################################################################################################
# Main program flow

if __name__=='__main__':
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	classifier = Classifier()
	DL_model = classifier.prepare_to_finetune(learning_rate) 

	if Method_type==0:
		X_train = np.load('Main_Data.npy',encoding='bytes')
		Y_train = np.load('Main_Labels.npy',encoding='bytes')
		X_test = np.load('X_Test.npy',encoding='bytes')
		Y_test = np.load('Y_Test.npy',encoding='bytes')

		if K.image_data_format()=='channels_first':
			X_train = np.reshape(X_train,(Availble_training_data_size,3,Img_resol,Img_resol))
			X_test = np.reshape(X_test,(Availble_test_data_size,3,Img_resol,Img_resol))
		else:
			X_train = np.reshape(X_train,(Availble_training_data_size,Img_resol,Img_resol,3))
			X_test = np.reshape(X_test,(Availble_test_data_size,Img_resol,Img_resol,3))				

		print ('training_size:{}'.format(X_train.shape))
		checkpointer = ModelCheckpoint(filepath= 'Model_save.hdf5', monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
		callbacks = [checkpointer]
		print ('###########################################################')
		print ('Evaluating baseline performance...')
		M = DL_model.fit(X_train,Y_train,epochs=Epochs,shuffle=True,batch_size=Batch_size, validation_data=(X_test,Y_test), verbose=1, callbacks=callbacks)
		print ('Baseline accuracy calculated:{}'.format(np.amax(np.array(M.history['val_acc']))))

	else:
		if Create_initial_dataset==True:
			X_oracle = np.load('Main_Data.npy',encoding='bytes')
			Y_oracle = np.load('Main_Labels.npy',encoding='bytes')

			X_test = np.load('X_Test.npy',encoding='bytes')
			Y_test = np.load('Y_Test.npy',encoding='bytes')

			X_oracle = np.reshape(X_oracle,(Availble_training_data_size,Img_resol,Img_resol,3))
			if Load_descriptors==False: 
				X_oracle_descriptor = calculate_descriptors(X_oracle)
				np.save('Oracle_descriptors.npy',X_oracle_descriptor)
				print ('Descriptor array saved!!')
			else:
				X_oracle_descriptor = np.load('Oracle_descriptors.npy',encoding='bytes')			

			X_train,Y_train = create_trainset() 

			if K.image_data_format()=='channels_last':        
				X_train = np.reshape(X_train,(Initial_dataset_size,Img_resol,Img_resol,3))
				X_oracle = np.reshape(X_oracle,((Availble_training_data_size - Initial_dataset_size),Img_resol,Img_resol,3))
				X_test = np.reshape(X_test,(len(X_test),Img_resol,Img_resol,3))

			else:
				X_train = np.reshape(X_train,(Initial_dataset_size,3,Img_resol,Img_resol))
				X_oracle = np.reshape(X_oracle,((Availble_training_data_size - Initial_dataset_size),3,Img_resol,Img_resol))
				X_test = np.reshape(X_test,(len(X_test),3,Img_resol,Img_resol))

			np.save('X_train.npy',X_train)
			np.save('Y_train.npy',Y_train)
			np.save('X_oracle.npy',X_oracle)
			np.save('Y_oracle.npy',Y_oracle)

			Test_Accuracy = []
			AUC_score = []
			current_AL_iteration = 0
			Class_count = []

		else:
			X_train = np.load('X_train.npy',encoding='bytes')
			Y_train = np.load('Y_train.npy',encoding='bytes')
			X_oracle = np.load('X_oracle.npy',encoding='bytes')
			Y_oracle = np.load('Y_oracle.npy',encoding='bytes')
			X_test = np.load('X_Test.npy',encoding='bytes')
			Y_test = np.load('Y_Test.npy',encoding='bytes')

			Test_Accuracy = np.load(Accuracy_array_name,encoding='bytes').tolist()
			AUC_score = np.load(AUC_array_name,encoding='bytes').tolist()
			current_AL_iteration = np.load('Current_AL_iteration.npy',encoding='bytes')[0]
			Class_count = list(np.load('Class_count.npy',encoding='bytes'))

			if K.image_data_format()=='channels_last':
				X_train = np.reshape(X_train,(len(X_train),Img_resol,Img_resol,3))
				X_oracle = np.reshape(X_oracle,(len(X_oracle),Img_resol,Img_resol,3))
				X_test = np.reshape(X_test,(len(X_test),Img_resol,Img_resol,3))	
			else:
				X_train = np.reshape(X_train,(len(X_train),3,Img_resol,Img_resol))
				X_oracle = np.reshape(X_oracle,(len(X_oracle),3,Img_resol,Img_resol))
				X_test = np.reshape(X_test,(len(X_test),3,Img_resol,Img_resol))	

		Inital_weights = DL_model.get_weights()
		checkpointer = ModelCheckpoint(filepath= 'Model_save.hdf5', monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=True)
		callbacks = [checkpointer]

		print ('train length:{}'.format(X_train.shape))
		for i in range(current_AL_iteration+1,AL_iterations+1):
			print ('############################################################')
			print ('Performing Active learning iteration {} for method {}'.format(i,Method_type))
			print ('Set sizes used for current AL iteration:')
			print ('Training set size:{}'.format(len(X_train)))
			print ('Oracle set size:{}'.format(len(X_oracle)))

			Positive_count = list(Y_train).count(1)
			Negative_count = list(Y_train).count(0)

			print ('Positive Class count:{}'.format(Positive_count))
			print ('Negative Class count:{}'.format(Negative_count))

			Class_count.append([Positive_count,Negative_count])

			M = DL_model.fit(X_train,Y_train,epochs=Epochs,shuffle=True,batch_size=Batch_size, validation_data=(X_test,Y_test), verbose=1, callbacks=callbacks)
			Test_Accuracy.append(np.amax(np.array(M.history['val_acc'])))
	
			print ('Calculating model predictions on Test set...')
			test_preds = DL_model.predict(X_test,batch_size=32,verbose=1)
			AUC_score.append(roc_auc_score(Y_test,test_preds))        

			np.save(Accuracy_array_name,np.array(Test_Accuracy))        
			np.save(AUC_array_name,np.array(AUC_score))
			np.save('Current_AL_iteration.npy',np.array([i]))
			np.save('Class_count.npy',np.asarray(Class_count))

			print ('Best Test accuracy achieved for this AL iteration:{0}\n AUC score:{1}'.format(Test_Accuracy[-1],AUC_score[-1]))
		
			Active_learning()

			if train_from_stratch==True:
				DL_model.set_weights(Inital_weights)

#############################################################################################################################################################33			

    
