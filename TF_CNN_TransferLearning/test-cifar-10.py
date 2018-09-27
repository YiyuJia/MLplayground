import matplotlib
matplotlib.use('Agg')


import numpy as np
import pickle
import skimage
import skimage.io
import skimage.transform


import pylab


import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


#Get batches
def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

#mapping label index to label name
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#deisplay prediction results (match or not?)

def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axs = pylab.subplots(10, 2, figsize=(12, 24))
    #fig = pylab.figure()
    #axs = fig.add_subplot(1, 1, 1)
    margin = 0.05
    ind = np.arange(n_classes)
    width = (1. - 2. * margin) / n_classes

    for image_i, (feature, label_id, prediction) in enumerate(zip(features, label_ids, predictions)):
        correct_name = label_names[label_id]
        pred_name = label_names[np.argmax(prediction)]

        is_match = 'False'

        if np.argmax(prediction) == label_id:
            is_match = 'True'

        predictions_array = []
        pred_names = []

        for index, pred_value in enumerate(prediction):
            tmp_pred_name = label_names[index]
            predictions_array.append({tmp_pred_name: pred_value})
            pred_names.append(tmp_pred_name)

        print('[{}] ground truth: {}, predicted result: {} | {}'.format(image_i, correct_name, pred_name, is_match))
        print('\t- {}\n'.format(predictions_array))

        #         print('image_i: ', image_i)
        #         print('axs: ', axs, ', axs len: ', len(axs))
        axs[image_i][0].imshow(feature)
        axs[image_i][0].set_title(pred_name)
        axs[image_i][0].set_axis_off()

        axs[image_i][1].barh(ind + margin, prediction, width)
        axs[image_i][1].set_yticks(ind + margin)
        axs[image_i][1].set_yticklabels(pred_names)

    pylab.savefig('predictResult.png')
    #pylab.show()
    #plt.tight_layout()


#Load dataset for testing
test_features, test_labels = pickle.load(open('preprocess_testing.p', mode='rb'))
tmpFeatures = []

for feature in test_features:
    tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    tmpFeatures.append(tmpFeature)

tmpFeatures = np.asarray(tmpFeatures)


#Testing
import random



config = tf.ConfigProto(allow_soft_placement=True)

#https://theqihao.github.io/2017/08/03/TensorFlow%E6%98%BE%E5%AD%98%E8%AE%BE%E7%BD%AE/
# "Best-fit with coalescing" algorithm for memory allocation
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.80

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config = config)


save_model_path = './image_classification'
batch_size = 64
n_samples = 10
top_n_predictions = 5


def test_model(tmpFeatures):
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph, config=config) as sess:
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

        for train_feature_batch, train_label_batch in batch_features_labels(tmpFeatures, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))

        # Print Random Samples
        random_test_features, random_test_labels = tuple(
            zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))

        tmpTestFeatures = []

        for feature in random_test_features:
            tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
            tmpTestFeatures.append(tmpFeature)

        random_test_predictions = sess.run(
            tf.nn.softmax(loaded_logits),
            feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels})

        display_image_predictions(random_test_features, random_test_labels, random_test_predictions)


test_model(tmpFeatures)

