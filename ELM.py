import numpy as np

x_train22=np.zeros((len(x_train),50))
for i in range(0,len(x_train)):
        x_train22[i,:]=x_train22[i]

x_test22=np.zeros((len(x_test),50))
for i in range(0,len(x_test)):
        x_test22[i,:]=x_test22[i]


from keras.utils import to_categorical


train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)


elm = ELM(x_test22.shape[1], train_Y_one_hot.shape[1])

elm.train(x_train22, train_Y_one_hot, "LOO")


Y = elm.predict(x_train22)

yy=np.argmax(Y,axis=1)

actual=np.argmax(train_Y_one_hot,axis=1)

from sklearn import metrics

error_elm=metrics.accuracy_score(yy,actual) * 10


acc_elm=100- error_elm

print("-------------------------------------")
print("     PERFORMANCE -------> (ELM)      ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_elm,'%')
print()
print("2. Error Rate =",error_elm)
    
