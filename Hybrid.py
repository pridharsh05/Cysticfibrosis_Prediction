# HYBRID

print("-------------------------------------")
print("     Hybrid ELM and Resnet -1D    ")
print("-------------------------------------")
print()



model = ResNet1D(num_blocks=3, filters=64, kernel_size=3)

# Create a ResNet1D model
from tensorflow.keras.models
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential()

model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(24,1)))


model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

res=model.evaluate(x_test, y_test)

# Make predictions on some new data

predictions = model.predict(x_train)


# ELM 


import numpy as np

x_train22=np.zeros((len(x_train),50))
for i in range(0,len(x_train)):
        x_train22[i,:]=x_train22[i]

x_test22=np.zeros((len(x_test),50))
for i in range(0,len(x_test)):
        x_test22[i,:]=x_test22[i]

from hpelm import ELM
from keras.utils import to_categorical


train_Y_one_hot = to_categorical(predictions)
test_Y_one_hot = to_categorical(y_test)


elm = ELM(x_test22.shape[1], train_Y_one_hot.shape[1])

elm.add_neurons(20, "sigm")
elm.add_neurons(10, "rbf_l2")
elm.train(x_train22, train_Y_one_hot, "LOO")


Y = elm.predict(x_train22)

yy=np.argmax(Y,axis=1)

actual=np.argmax(train_Y_one_hot,axis=1)

from sklearn import metrics

error_elm=metrics.accuracy_score(yy,actual)


acc_hyb = 100- error_elm

print("----------------------------------------")
print("     PERFORMANCE -------> (Hybrid)      ")
print("----------------------------------------")
print()
print("1. Accuracy   =", acc_hyb,'%')
print()
print("2. Error Rate =",error_elm)

# COMPARISON GRAPH 


import seaborn as sns 
sns.barplot(x=['Resnet- 1D','ELM','Hybrid'],y=[acc_resnet,acc_elm,acc_hyb])
plt.show()

#Count of Cancer
    
sns.countplot(x ='y', data = df)
plt.title("Count of Cancer")
plt.show()    
