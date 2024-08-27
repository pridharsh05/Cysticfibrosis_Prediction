print("----------------------------------------")
print("CLASSIFICATION")
print("------------------------------------")
print()    
 

 
import tensorflow as tf

class ResNetBlock1D(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, shortcut=True):
        super(ResNetBlock1D, self).__init__()

        self.shortcut = shortcut

        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if self.shortcut and strides == 1 and filters == 1:
            self.shortcut_conv = tf.identity
        else:
            self.shortcut_conv = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding='same'),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut_conv(inputs)

        x = x + shortcut

        return x

class ResNet1D(tf.keras.Model):
    def __init__(self, num_blocks, filters, kernel_size, num_classes=2):
        super(ResNet1D, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.blocks = tf.keras.Sequential([
            ResNetBlock1D(filters, kernel_size)
            for _ in range(num_blocks)
        ])

       

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.blocks(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

# Example usage:

model = ResNet1D(num_blocks=3, filters=64, kernel_size=3)

# Create a ResNet1D model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

model = Sequential()

model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(24,1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# Compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on some validation data

res=model.evaluate(x_test, y_test)

# Make predictions on some new data

predictions = model.predict(x_test)

print("-------------------------------------")
print("     Performance ---> RESNET 1D      ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_resnet,'%')
print()
print("2. Error Rate =",error)
