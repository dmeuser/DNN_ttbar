import numpy as np
import tensorflow as tf
from tensorflow import keras


class EarlyStoppingCombined(keras.callbacks.Callback):
	"""Stop training when the validation loss deviaties from the training loss by a certain percentage OR when the validation loss stops decreasing
	
	Arguments:
		patience: Number of eopchs to wait after min has been hit. 
		percentage: percentage deviation between training- and validation loss that may be reached
	"""
	def __init__(self, patience=0, percentage=2, percentagePatience=0, generalizationPatience=0):
		super(EarlyStoppingCombined, self).__init__()
		self.patience = patience
		self.percentage = percentage
		self.percentagePatience = percentagePatience
		self.generalizationPatience = generalizationPatience
		self.best_weights = None
		self.best_epoch = 0

	def on_train_begin(self, logs=None):
		self.wait=0
		self.stopped_epoch=0
		self.isOverThreshold = False
		self.genErrWait = 0
		self.genErr = np.Inf
		self.best = np.Inf

	def on_epoch_end(self, epoch, logs=None):
		currLoss_train = logs.get("loss")
		currLoss_val = logs.get("val_loss")
		#  ~currLoss_train = logs.get("logcosh")
		#  ~currLoss_val = logs.get("val_logcosh")
		
		if epoch >= 10 and np.abs(currLoss_val - currLoss_train)/currLoss_train > float(self.percentage)/100:
			self.percentageWait += 1
		else: self.percentageWait = 0
		
		if np.less(self.genErr, np.abs(currLoss_val - currLoss_train)):
			self.genErrWait += 1
		else: self.genErrWait = 0
		self.genErr = np.abs(currLoss_val - currLoss_train)
			
		if np.less(currLoss_val, self.best):
			self.best = currLoss_val
			self.wait = 0
			self.best_weights = self.model.get_weights()
			self.best_epoch = epoch
		else:
			self.wait += 1
			print(self.wait)
		if self.wait >= self.patience or self.isOverThreshold or self.genErrWait >= self.generalizationPatience:
			self.stopped_epoch = epoch
			self.model.stop_training = True
			print("Restoring model weights from the end of the best epoch (epoch {}).".format(self.best_epoch+1))
			self.model.set_weights(self.best_weights)

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			if self.wait >= self.patience:
				print("\nEarly stopped in Epoch {} due to stagnating progress\n".format(self.stopped_epoch+1))
			if self.isOverThreshold:
				print("\nEarly stopped in Epoch {} due to generalization error over {}%\n".format(self.stopped_epoch+1, self.percentage))
			if self.genErrWait >= self.generalizationPatience:
				print("\nEarly stopped in Epoch {} due to increasing generalization error for {} epochs\n".format(self.stopped_epoch+1, self.genErrWait))

