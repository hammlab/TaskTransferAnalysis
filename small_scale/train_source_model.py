import tensorflow as tf
import numpy as np
from utils import eval_accuracy_and_loss, load_data
from models import shared_model, classification_model
import argparse

print("################ START ################")
parser = argparse.ArgumentParser(description='Train source model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='MNIST', choices=['MNIST', 'KMNIST', 'FMNIST', 'USPS', 'EMNIST'], help='Name of SRC data')
args = parser.parse_args()

SRC = args.SRC

NUM_CLASSES_MAIN = 26 if SRC == 'EMNIST' else 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 1
EPOCHS = 31
BATCH_SIZE = 256
REP_DIM = 200
CHECKPOINT_PATH = "./checkpoints/vanilla_"+str(REP_DIM)+"_" + SRC

SRC_x_train, SRC_y_train, SRC_x_test, SRC_y_test = load_data(SRC)

indices = np.arange(len(SRC_x_train))
np.random.shuffle(indices)
SRC_x_train = np.array(SRC_x_train[indices], np.float32)
SRC_y_train = np.array(SRC_y_train[indices], np.float32)

train_labels = tf.keras.utils.to_categorical(SRC_y_train, NUM_CLASSES_MAIN)
test_labels = tf.keras.utils.to_categorical(SRC_y_test, NUM_CLASSES_MAIN)

SRC_x_train = np.array(SRC_x_train/255., np.float32)
SRC_x_test = np.array(SRC_x_test/255., np.float32)

train_images = SRC_x_train.reshape((SRC_x_train.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))
test_images = SRC_x_test.reshape((SRC_x_test.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))

ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

shared = shared_model([50000, IMG_HEIGHT, IMG_WIDTH, NCH], output_dim=REP_DIM)
dummy_classifier = classification_model(shared, NUM_CLASSES_MAIN, input_dim=REP_DIM)

optimizer_shared = tf.keras.optimizers.Adam(1E-3, beta_1=0.5, decay=5E-4)
optimizer_dummy_classifier = tf.keras.optimizers.Adam(1E-3, beta_1=0.5, decay=5E-4)

ckpt = tf.train.Checkpoint(shared = shared, classifier=dummy_classifier)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

@tf.function
def train_erm(data, class_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        outputs = dummy_classifier(shared(data, training=True), training=True)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs))

    gradients_model = tape.gradient(loss, shared.trainable_variables)
    gradients_dummy_classifier = tape.gradient(loss, dummy_classifier.trainable_variables)
    
    optimizer_shared.apply_gradients(zip(gradients_model, shared.trainable_variables))
    optimizer_dummy_classifier.apply_gradients(zip(gradients_dummy_classifier, dummy_classifier.trainable_variables))
 

for epoch in range(EPOCHS):
    
    nb_batches_train = int(len(train_images)/BATCH_SIZE)
    ind_shuf = np.arange(len(train_images))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE*batch, min(BATCH_SIZE*(1+batch), len(train_images)))
        
        xs = train_images[ind_batch]
        ys = train_labels[ind_batch]
        
        train_erm(xs, ys)
        
    if epoch % 10 == 0:
        test_accuracy = eval_accuracy_and_loss(test_images, test_labels, shared, dummy_classifier)
        print("Epoch:", epoch)
        print("Test:", test_accuracy[0], test_accuracy[1])
        ckpt_model_save_path = ckpt_manager.save()
        print("\n")