import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_lables) = model.load_data()

print(training_images[0])
# plt.imshow(training_images[0])
plt.show()


training_labels =training_labels/255
training_images = training_images /255

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation =tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation =tf.nn.softmax)])

model.compile(optimizer= tf.train.AdamOptimizer(),
              loss= 'spase_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
