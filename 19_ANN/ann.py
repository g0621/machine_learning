import pandas as pd,numpy as np

data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:,3:13].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_enc_x1 = LabelEncoder()
X[:, 1] = label_enc_x1.fit_transform(X[:,1])
label_enc_x2 = LabelEncoder()
X[:, 2] = label_enc_x1.fit_transform(X[:,2])

ohe = OneHotEncoder(categorical_features=[1])
X = ohe.fit_transform(X).toarray()
# excluding first to eliminate dummy variable trap
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)  # as we don't want to know the values for test

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(6, input_shape=(11,), use_bias=True,kernel_initializer='random_uniform',activation='relu'))

classifier.add(Dense(6, use_bias=True,kernel_initializer='random_uniform',activation='relu'))
classifier.add(Dense(1, use_bias=True,kernel_initializer='random_uniform',activation='sigmoid'))

# if more than two output then categorical_crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(X_train, y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(X_test)
y_pred = ( y_pred > 0.5 )

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)