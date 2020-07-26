from tensorflow.keras.datasets import mnist

(x_train, y_train), (_,_) = mnist.load_data()

print(x_train.shape)
print(type(x_train))