import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

from Approximator import Approximator
from util import DotDic

class OXONeuralNetwork(Approximator):

	def __init__(self, task):
		self.args = DotDic({
			'numEpochs': 30,
			'batchSize': 64,
			'numChannels': 512,
			'dropoutRate': 0.4,
			'lr': 1e-4,
			'checkpoint_path': './tasks/OXO/OXOmodel.h5'
		})
		self.task = task
		self.model = self._setup_model()

	def _setup_model(self):
		state_shape = self.task.get_state_shape()
		x = Input(shape=state_shape)
		x_reshaped = Reshape(state_shape + (1,))(x)
		c1 = BatchNormalization()(Conv2D(self.args.numChannels, 3, padding='same', activation='relu')(x_reshaped))
		c2 = BatchNormalization()(Conv2D(self.args.numChannels, 3, padding='same', activation='relu')(c1))
		c3 = BatchNormalization()(Conv2D(self.args.numChannels/2, 3, padding='same', activation='relu')(c2))
		c4 = BatchNormalization()(Conv2D(self.args.numChannels/2, 3, padding='same', activation='relu')(c3))
		c4_flat = Flatten()(c4)
		d1 = Dropout(self.args.dropoutRate)(BatchNormalization()(Dense(1024)(c4_flat)))
		d2 = Dropout(self.args.dropoutRate)(BatchNormalization()(Dense(512)(d1)))
		p = Dense(self.task.get_num_actions(), activation='softmax', name='p')(d2)
		v = Dense(1, activation='tanh', name='v')(d2)
		
		model = Model(inputs=x, outputs=[p, v])
		model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=RMSprop(lr=self.args.lr))
		return model

	def train(self, examples):
		args = self.args
		states, policies, values = tuple(zip(*examples))
		states = np.asarray(states)
		policies = np.asarray(policies)
		values = np.asarray(values)

		callbacks_list = [
			EarlyStopping(
				monitor='v_loss',
				patience=1,
			),
			ModelCheckpoint(
				filepath=self.args.checkpoint_path,
				monitor='v_loss',
				save_best_only=True,
			),
			ReduceLROnPlateau(monitor='v_loss'),
		]

		self.model.fit(
			x=states, 
			y=[policies, values],
			batch_size=args.batch_size, 
			epochs=args.numEpochs,
			callbacks=callbacks_list)
		self.model.load_weights(self.args.checkpoint_path)

	def predict(self, s):
		p, v = self.model.predict(np.expand_dims(s, 0))
		return p[0], v[0]