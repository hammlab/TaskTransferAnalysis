import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
sys.path.append("../")
import os, sys
import tensorflow as tf
import numpy as np
from utils import eval_accuracy_and_loss_cls, load_data, get_all_reps
from models import shared_model, classification_model
import argparse
from sklearn.preprocessing import StandardScaler

print("################ START ################")
parser = argparse.ArgumentParser(description='Finetune from MNIST', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='MNIST', choices=['MNIST', 'KMNIST', 'FMNIST', 'USPS', 'EMNIST'], help='Name of SRC data')
parser.add_argument('--TRG', type=str, default='USPS', choices=['MNIST', 'KMNIST', 'FMNIST', 'USPS', 'EMNIST'], help='Name of TRG data')
parser.add_argument('--SEED', type=int, default=2023, help='seed')
parser.add_argument('--TAU', type=float, default=0.2, help='TAU')
args = parser.parse_args()

SRC = args.SRC
TRG = args.TRG
np.random.seed(args.SEED)

if TRG == 'EMNIST':
    sys.exit('EMNIST cannot be used as Target data')

SRC_NUM_CLASSES = 26 if SRC == 'EMNIST' else 10
TRG_NUM_CLASSES = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 1
REP_DIM = 200

if TRG == 'USPS':
    EPOCHS = 1001
    BATCH_SIZE = 100
else:
    EPOCHS = 501
    BATCH_SIZE = 500
    

TAU = float(args.TAU)
CHECKPOINT_PATH = "./checkpoints/vanilla_200_"+SRC

if not os.path.exists(CHECKPOINT_PATH):
    sys.exit("No Model:"+CHECKPOINT_PATH)

###############
SRC_x_train, SRC_y_train, SRC_x_test, SRC_y_test = load_data(SRC)

SRC_x_train = SRC_x_train.reshape((SRC_x_train.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))
SRC_x_test = SRC_x_test.reshape((SRC_x_test.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))

src_idx_train = np.arange(len(SRC_x_train))
src_idx_test = np.arange(len(SRC_x_test))
np.random.shuffle(src_idx_train)
np.random.shuffle(src_idx_test)

source_x_train, source_y_train = SRC_x_train[src_idx_train], SRC_y_train[src_idx_train]
source_x_test, source_y_test = SRC_x_test[src_idx_test], SRC_y_test[src_idx_test]

source_x_train = np.float32(source_x_train) / 255.
source_y_train = tf.keras.utils.to_categorical(source_y_train, SRC_NUM_CLASSES)

source_x_test = np.float32(source_x_test) / 255.
source_y_test = tf.keras.utils.to_categorical(source_y_test, SRC_NUM_CLASSES)
source_x_test, source_y_test = source_x_test, source_y_test
###############

###############
TRG_x_train, TRG_y_train, TRG_x_test, TRG_y_test = load_data(TRG)

TRG_x_train = TRG_x_train.reshape((TRG_x_train.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))
TRG_x_test = TRG_x_test.reshape((TRG_x_test.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))

trg_idx_train = np.arange(len(TRG_x_train))
trg_idx_test = np.arange(len(TRG_x_test))
np.random.shuffle(trg_idx_train)
np.random.shuffle(trg_idx_test)

target_x_train, target_y_train = TRG_x_train[trg_idx_train], TRG_y_train[trg_idx_train]
target_x_test, target_y_test = TRG_x_test[trg_idx_test], TRG_y_test[trg_idx_test]

target_y_train = tf.keras.utils.to_categorical(target_y_train, TRG_NUM_CLASSES)
target_x_train = np.float32(target_x_train) / 255.

target_y_test = tf.keras.utils.to_categorical(target_y_test, TRG_NUM_CLASSES)
target_x_test = np.float32(target_x_test) / 255.
target_x_test, target_y_test = target_x_test, target_y_test
###############

shared = shared_model([50000, IMG_HEIGHT, IMG_WIDTH, NCH])
classifier_source = classification_model(shared, SRC_NUM_CLASSES)
classifier_target = classification_model(shared, TRG_NUM_CLASSES)

if SRC == 'USPS':
    optimizer_classifier_target = tf.keras.optimizers.Adam(1E-3, beta_1=0.5)
else:
    optimizer_classifier_target = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

ckpt = tf.train.Checkpoint(shared = shared, classifier = classifier_source)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=1) 
ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

ckpt_trg = tf.train.Checkpoint(classifier_target = classifier_target)

source_train_rep = get_all_reps(source_x_train, shared)
source_test_rep = get_all_reps(source_x_test, shared)
target_train_rep = get_all_reps(target_x_train, shared)
target_test_rep = get_all_reps(target_x_test, shared)

src_scaler = StandardScaler()
trg_scaler = StandardScaler()

src_scaler.fit(source_train_rep)
trg_scaler.fit(target_train_rep)

source_train_rep = src_scaler.transform(source_train_rep)
source_test_rep = src_scaler.transform(source_test_rep)

target_train_rep = trg_scaler.transform(target_train_rep)
target_test_rep = trg_scaler.transform(target_test_rep)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def gradient_penalty(data, class_labels, random_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        data = tf.convert_to_tensor(data, tf.float32)
        tape.watch(data)
        outputs = classifier_target(data, training=True)
        loss = ce_loss_none(class_labels, outputs) + ce_loss_none(random_labels, outputs)
     
    gradients_data_var = tape.gradient(loss, data)  
    g_norm2 = tf.sqrt(tf.reduce_sum(gradients_data_var ** 2, axis=[1]))
    grad_penalty = tf.reduce_mean((tf.nn.relu(g_norm2 - TAU) ** 2))
    max_grad_norm = tf.reduce_max(g_norm2)
    
    return grad_penalty, max_grad_norm
    

@tf.function
def train_erm(data_rep, class_labels, random_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        outputs = classifier_target(data_rep, training=True)
        grad_penalty, _ = gradient_penalty(data_rep, class_labels, random_labels)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs)) + 1E5 * grad_penalty

    gradients_classifier_target = tape.gradient(loss, classifier_target.trainable_variables)
    optimizer_classifier_target.apply_gradients(zip(gradients_classifier_target, classifier_target.trainable_variables)) 
    return grad_penalty
    
results = []
for epoch in range(EPOCHS):
    
    nb_batches_train = int(len(target_train_rep)/BATCH_SIZE)
    ind_shuf = np.arange(len(target_train_rep))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE*batch, min(BATCH_SIZE*(1+batch), len(target_train_rep)))
        
        xs = target_train_rep[ind_batch]
        ys = target_y_train[ind_batch]
        ys_random = tf.keras.utils.to_categorical(np.random.randint(0, TRG_NUM_CLASSES, len(xs)), TRG_NUM_CLASSES)
        
        gp = train_erm(xs, ys, ys_random)
        
    if epoch % 100 == 0:
        test_accuracy = eval_accuracy_and_loss_cls(target_test_rep, target_y_test, classifier_target)
        print("Epoch:", epoch)
        op = gradient_penalty(target_test_rep, target_y_test, tf.keras.utils.to_categorical(np.random.randint(0, TRG_NUM_CLASSES, len(target_test_rep)), TRG_NUM_CLASSES))
        print("Record SRC:", SRC, "TRG:", TRG, "Test: ", test_accuracy[0], test_accuracy[1], op[1].numpy())
        print("\n")

print("################ END ################")