import tensorflow as tf 
import cv2 
import os 
import numpy as np 
import matplotlib.pyplot as plt

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory, d))]
    
    labels = []
    images = []
    
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
        for f in os.listdir(label_directory)
        if f.endswith(".png")]

        
        for f in file_names:
            images.append(cv2.imread(f))
            labels.append(int(d))
    return images, labels


def neural_net(x):
     layer_1 = tf.layers.dense(x, 450, activation = tf.nn.relu)
     layer_2 = tf.layers.dense(layer_1, 325, activation = tf.nn.relu)
     out_layer = tf.layers.dense(layer_2, 125)
     return out_layer

ROOT_PATH = "C:/"

train_data_directory = os.path.join(ROOT_PATH, "datta")

images, labels = load_data(train_data_directory)

print("ukuran gambar: ", images[0].shape)
print("Banyaknya label: ", len(set(labels)))

dim = (30,30)

images30 = []

for image in images:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    images30.append(cv2.resize(gray_image, dim, interpolation = cv2.INTER_CUBIC))

                                 
x = tf.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tf.placeholder(dtype = tf.int32, shape = [None])
   
images_flat = tf.contrib.layers.flatten(x)
 
logits = neural_net(images_flat)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y))
 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(4) 

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer()) 
    for i in range(250): 
        _, loss_value, acc = sess.run([train_op,loss,accuracy],feed_dict={x:images30, y:labels})

        if i%10==0: 
            print("Loss: ", loss_value, "Accuracy: ", acc) 
