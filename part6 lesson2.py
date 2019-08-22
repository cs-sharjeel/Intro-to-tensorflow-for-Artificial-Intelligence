import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_lables), (test_images, test_lables) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images= training_images/255
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255

