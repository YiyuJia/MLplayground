
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import argparse
import tarfile
import pickle
import numpy as np

import skimage
import skimage.io
import skimage.transform

import tensorflow as tf
import tensornets as nets


parser = argparse.ArgumentParser()
parser.add_argument("-e", help="specify epoch number", default=7, type=int)
parser.add_argument("-b", help="specify batch size", default=32, type=int)

args = parser.parse_args()



#config = tf.ConfigProto(allow_soft_placement=True)

# "Best-fit with coalescing" algorithm for memory allocation
#config.gpu_options.allocator_type = 'BFC'
#config.gpu_options.per_process_gpu_memory_fraction = 0.80

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#download cifar-10 dataset
cifar10_dataset_folder_path = 'cifar-10-batches-py'

class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

""" 
    check if the data (zip) file is already downloaded
    if not, download it from "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" and save as cifar-10-python.tar.gz
"""
if not isfile('cifar-10-python.tar.gz'):
    with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()

#load raw data and reshape the input images
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

#one hot encoding method for label data
def one_hot_encode(x):
    """
        argument
            - x: a list of labels
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    encoded = np.zeros((len(x), 10))

    for idx, val in enumerate(x):
        encoded[idx][val] = 1

    return encoded


#save the modified input(feature) and labels
def _preprocess_and_save(one_hot_encode, features, labels, filename):
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        # preprocess the 90% of the whole dataset of the batch
        # - normalize the features
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        _preprocess_and_save(one_hot_encode,
                             features[:-index_of_validation], labels[:-index_of_validation],
                             'preprocess_batch_' + str(batch_i) + '.p')

        # unlike the training dataset, validation dataset will be added through all batch dataset
        # - take 10% of the whold dataset of the batch
        # - add them into a list of
        #   - valid_features
        #   - valid_labels
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         'preprocess_validation.p')

    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all testing data
    _preprocess_and_save(one_hot_encode,
                         np.array(test_features), np.array(test_labels),
                         'preprocess_testing.p')


preprocess_and_save_data(cifar10_dataset_folder_path, one_hot_encode)


#Traning the model for CIFAR-10
#Load data

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None, 10), name='output_y')

#Hyper-parameters
#You don't need many epochs since this is transfer learning.
#you can increase the batch size if you have a very high-end GPU running. 32 is the maximum value I could try with NVIDIA GTX 1080Ti
learning_rate = 0.00001
epochs = args.e
batch_size = args.b

#Load VGG19 model, Define loss, train, and accuracy tensor/Operation
#The first step is to predefined VGG19 model.
#
#if you want to run the model as it is, just pass the input placeholder.
#if you want to train on your own image dataset, set is_training=True, and classes=# of class
#nets.VGG19 returns the final layer of the VGG19 which is softmax. If you know, tensorflow comes with tf.nn.softmax_cross_entropy_with_logits function, and this applies softmax and cross entropy together. However, nets.VGG19 returns the layer already applied with softmax, so we need only cross entropy. That can be achieved by tf.losses.softmax_cross_entropy function.
#
#In your choice, you can choose your favorite optimizer. I am going to use Adam optimizer since it is known to work moderate for most of deep learning problems.

logits = nets.VGG19(x, is_training=True, classes=10)
model = tf.identity(logits, name='logits')

loss = tf.losses.softmax_cross_entropy(y, logits)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

logits.print_outputs()

logits.print_summary()

#Get batches
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


# The size of each CIFAR-10 image is 32x32, and VGG19 takes input image sizes 224x224 which is incompatible.
# Each CIFAR-10 image should be resized so that it can be fed into the VGG19 model.
# skimage.transform.resize function does the trick. It takes parameters..

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    tmpFeatures = []

    for feature in features:
        tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
        tmpFeatures.append(tmpFeature)

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(tmpFeatures, labels, batch_size)


#Get inputs for validation
tmpValidFeatures = []

for feature in valid_features:
    tmpValidFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    tmpValidFeatures.append(tmpValidFeature)

tmpValidFeatures = np.array(tmpValidFeatures)


print(tmpValidFeatures.shape)


#Training

save_model_path = './image_classification'

print('Training...')
with tf.Session(config=config) as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    print('global_variables_initializer ... done ...')
    sess.run(logits.pretrained())
    print('model.pretrained ... done ... ')

    # Training cycle
    print('starting training ... ')
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                sess.run(train, {x: batch_features, y: batch_labels})

            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')

            # calculate the mean accuracy over all validation dataset
            valid_acc = 0
            for batch_valid_features, batch_valid_labels in batch_features_labels(tmpValidFeatures, valid_labels,
                                                                                  batch_size):
                valid_acc += sess.run(accuracy, {x: batch_valid_features, y: batch_valid_labels})

            tmp_num = tmpValidFeatures.shape[0] / batch_size
            print('Validation Accuracy: {:.6f}'.format(valid_acc / tmp_num))

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)


















