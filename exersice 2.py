import tensorflow as tf
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.1):
            print("Reached Accuracy 99%, So stop training..")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.mnist
# plt.imshow(x_train[10])

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 25.0
y_train = y_train / 25.0

# plt.imshow(x_train[10])

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])