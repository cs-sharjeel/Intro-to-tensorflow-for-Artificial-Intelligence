import tensorflow as tf
import matplotlib .pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_lables), (test_images, test_lables) = mnist.load_data()

# print(training_lables[12])
# print(training_images[12])
plt.show()
# print("/n====================================================")
training_images = training_images/255
test_images = test_images/255
# print(training_images[12])
plt.show()

model =tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer= tf.compat.v1.train.AdamOptimizer(),
              loss = 'mean_squared_error',
              metrics = ['accuracy'])

model.fit(training_images,training_lables, epochs=5)

# model.evaluate(test_images, test_lables)

clas = model.predict(test_images)
print(clas[0])

print(test_lables[0])

