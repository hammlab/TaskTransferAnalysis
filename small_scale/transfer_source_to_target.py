import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
sys.path.append("../")
import os, sys
import tensorflow as tf
import numpy as np
from utils import get_all_reps, get_set_based_on_prior, get_weigted_SRC_loss, eval_accuracy_and_loss_cls_with_B, compute_WD_same_label_sets, load_data, eval_accuracy_and_loss_cls
from models import shared_model, classification_model
import ot
from scipy.spatial.distance import cdist 
import argparse
from sklearn.preprocessing import StandardScaler

print("################ START ################")
parser = argparse.ArgumentParser(description='Finetune from MNIST', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='MNIST', choices=['MNIST', 'KMNIST', 'FMNIST', 'USPS', 'EMNIST'], help='Name of SRC data')
parser.add_argument('--TRG', type=str, default='USPS', choices=['MNIST', 'KMNIST', 'FMNIST', 'USPS', 'EMNIST'], help='Name of TRG data')
parser.add_argument('--LAMDA_WD', type=float, default=2, help='Weight for WD term')
parser.add_argument('--BATCH_SIZE', type=int, default=500, help='Batch Size')
parser.add_argument('--SEED', type=int, default=2023, help='seed')
parser.add_argument('--TAU', type=float, default=0.2, help='TAU')

args = parser.parse_args()
np.random.seed(args.SEED)

SRC = args.SRC
TRG = args.TRG

if TRG == 'EMNIST':
    sys.exit('EMNIST cannot be used as Target data')

SRC_NUM_CLASSES = 26 if SRC == 'EMNIST' else 10
TRG_NUM_CLASSES = 10 
IMG_WIDTH = 28
IMG_HEIGHT = 28
NCH = 1
REP_DIM = 200
EPOCHS = 1001

if SRC == 'USPS':
    EPOCHS_GP = 2001
else:
    EPOCHS_GP = 501

LAMDA_WD = float(args.LAMDA_WD)
LAMDA_PM = 1
LAMDA_IC = 1

TAU = float(args.TAU)
CHECKPOINT_PATH = "./checkpoints/vanilla_200_"+SRC

LAMBDA = 1E8 
BATCH_SIZE = int(args.BATCH_SIZE)

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

source_x_train_all, source_y_train_all = SRC_x_train[src_idx_train], SRC_y_train[src_idx_train]
source_x_test_all, source_y_test_all = SRC_x_test[src_idx_test], SRC_y_test[src_idx_test]

source_x_train_all = np.float32(source_x_train_all) / 255.
source_y_train_all = tf.keras.utils.to_categorical(source_y_train_all, SRC_NUM_CLASSES)

source_x_test_all = np.float32(source_x_test_all) / 255.
source_y_test_all = tf.keras.utils.to_categorical(source_y_test_all, SRC_NUM_CLASSES)
###############

###############
TRG_x_train, TRG_y_train, TRG_x_test, TRG_y_test = load_data(TRG)

TRG_x_train = TRG_x_train.reshape((TRG_x_train.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))
TRG_x_test = TRG_x_test.reshape((TRG_x_test.shape[0], IMG_WIDTH, IMG_HEIGHT, NCH))

trg_idx_train = np.arange(len(TRG_x_train))
trg_idx_test = np.arange(len(TRG_x_test))
np.random.shuffle(trg_idx_train)
np.random.shuffle(trg_idx_test)

target_x_train_all, target_y_train_all = TRG_x_train[trg_idx_train], TRG_y_train[trg_idx_train]
target_x_test_all, target_y_test_all = TRG_x_test[trg_idx_test], TRG_y_test[trg_idx_test]

prior_target = np.zeros(TRG_NUM_CLASSES, np.float32)
for i in range(TRG_NUM_CLASSES):
    prior_target[i] = len(np.argwhere(target_y_train_all==i).flatten())/len(target_y_train_all)    

target_y_train_all = tf.keras.utils.to_categorical(target_y_train_all, TRG_NUM_CLASSES)
target_x_train_all = np.float32(target_x_train_all) / 255.

target_y_test_all = tf.keras.utils.to_categorical(target_y_test_all, TRG_NUM_CLASSES)
target_x_test_all = np.float32(target_x_test_all) / 255.
target_x_test_all, target_y_test_all = target_x_test_all, target_y_test_all
###############

shared = shared_model([50000, IMG_HEIGHT, IMG_WIDTH, NCH])

ckpt_model = tf.train.Checkpoint(shared = shared)
ckpt_manager_model = tf.train.CheckpointManager(ckpt_model, CHECKPOINT_PATH, max_to_keep=1) 
ckpt_model.restore(ckpt_manager_model.latest_checkpoint).expect_partial()

source_train_rep_all = get_all_reps(source_x_train_all, shared)
target_train_rep_all = get_all_reps(target_x_train_all, shared)
source_test_rep_all = get_all_reps(source_x_test_all, shared)
target_test_rep_all = get_all_reps(target_x_test_all, shared)

####################  Select data from TRG_NUM_CLASSES of SRC ###################
if SRC in ['EMNIST']:
    SELECTED_CLASSES = np.arange(SRC_NUM_CLASSES)
    np.random.shuffle(SELECTED_CLASSES)
    SELECTED_CLASSES = SELECTED_CLASSES[:TRG_NUM_CLASSES]
    
    for i in range(len(SELECTED_CLASSES)):
        if i == 0:
            train_mask = (np.argmax(source_y_train_all, 1) == SELECTED_CLASSES[i])
            test_mask = (np.argmax(source_y_test_all, 1) == SELECTED_CLASSES[i])
        else:
            train_mask = train_mask | (np.argmax(source_y_train_all, 1) == SELECTED_CLASSES[i])
            test_mask = test_mask | (np.argmax(source_y_test_all, 1) == SELECTED_CLASSES[i])
    
    selected_train_indices = np.argwhere(train_mask).flatten()
    selected_test_indices = np.argwhere(test_mask).flatten()
    
    SRC_NUM_CLASSES = len(SELECTED_CLASSES)
    
    source_train_rep_all = np.array(source_train_rep_all[selected_train_indices])
    source_y_train_all = np.array(source_y_train_all[selected_train_indices])
    
    source_test_rep_all = np.array(source_test_rep_all[selected_test_indices])
    source_y_test_all = np.array(source_y_test_all[selected_test_indices])
    
    source_y_train_all_new = np.zeros(len(source_y_train_all))
    source_y_test_all_new = np.zeros(len(source_y_test_all))
    
    for i in range(len(SELECTED_CLASSES)):
        idx_train = np.argwhere(np.argmax(source_y_train_all, 1) == SELECTED_CLASSES[i]).flatten()
        idx_test = np.argwhere(np.argmax(source_y_test_all, 1) == SELECTED_CLASSES[i]).flatten()
        
        source_y_train_all_new[idx_train] = i
        source_y_test_all_new[idx_test] = i
        
    source_y_train_all_new = tf.keras.utils.to_categorical(source_y_train_all_new, SRC_NUM_CLASSES)
    source_y_test_all_new = tf.keras.utils.to_categorical(source_y_test_all_new, SRC_NUM_CLASSES)
    
    source_y_train_all = np.array(source_y_train_all_new)
    source_y_test_all = np.array(source_y_test_all_new)
    
prior_source = np.ones(SRC_NUM_CLASSES, np.float32)/SRC_NUM_CLASSES

src_scaler = StandardScaler()
trg_scaler = StandardScaler()

src_scaler.fit(source_train_rep_all)
trg_scaler.fit(target_train_rep_all)

source_train_rep_all = src_scaler.transform(source_train_rep_all)
source_test_rep_all = src_scaler.transform(source_test_rep_all)

target_train_rep_all = trg_scaler.transform(target_train_rep_all)
target_test_rep_all = trg_scaler.transform(target_test_rep_all)


################ Train with gradient Penalty
classifier = classification_model(shared, SRC_NUM_CLASSES)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

if SRC == 'USPS':
    optimizer_classifier = tf.keras.optimizers.Adam(5E-3, beta_1=0.5)
else:
    optimizer_classifier = tf.keras.optimizers.Adam(1E-4, beta_1=0.5)

def gradient_penalty(data, class_labels, random_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        data = tf.convert_to_tensor(data, tf.float32)
        tape.watch(data)
        outputs = classifier(data, training=True)
        loss = ce_loss_none(class_labels, outputs) + ce_loss_none(random_labels, outputs)
     
    gradients_data_var = tape.gradient(loss, data)  
    g_norm2 = tf.sqrt(tf.reduce_sum(gradients_data_var ** 2, axis=[1]))
    grad_penalty = tf.reduce_mean((tf.nn.relu(g_norm2 - TAU) ** 2))
    max_grad_norm = tf.reduce_max(g_norm2)
    
    return grad_penalty, max_grad_norm
    
 
@tf.function
def train_cls_with_grad_penalty(data, class_labels, random_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        outputs = classifier(data, training=True)
        grad_penalty, _ = gradient_penalty(data, class_labels, random_labels)
        loss = tf.reduce_mean(ce_loss_none(class_labels, outputs)) + 1E10 * grad_penalty

    gradients_classifier = tape.gradient(loss, classifier.trainable_variables)
    optimizer_classifier.apply_gradients(zip(gradients_classifier, classifier.trainable_variables))
    return grad_penalty
    

results = []
for epoch in range(EPOCHS_GP):
    
    nb_batches_train = int(len(source_train_rep_all)/BATCH_SIZE)
    ind_shuf = np.arange(len(source_train_rep_all))
    np.random.shuffle(ind_shuf)
    
    for batch in range(nb_batches_train):
        ind_batch = range(BATCH_SIZE*batch, min(BATCH_SIZE*(1+batch), len(source_train_rep_all)))
        
        xs = source_train_rep_all[ind_batch]
        ys = source_y_train_all[ind_batch]
        ys_random = tf.keras.utils.to_categorical(np.random.randint(0, SRC_NUM_CLASSES, len(xs)), SRC_NUM_CLASSES)
        
        train_cls_with_grad_penalty(xs, ys, ys_random)
        
    if epoch % 100 == 0:
        print(args)
        test_accuracy = eval_accuracy_and_loss_cls(source_test_rep_all, source_y_test_all, classifier)
        print("Epoch:", epoch)
        op = gradient_penalty(source_test_rep_all, source_y_test_all, tf.keras.utils.to_categorical(np.random.randint(0, SRC_NUM_CLASSES, len(source_test_rep_all)), SRC_NUM_CLASSES))
        print("Record SRC:", SRC, "TRG:", TRG, "Test: ", test_accuracy[0], test_accuracy[1], op[1].numpy())
        print("\n")


################ Setup Transformations
transformation_A = tf.Variable(tf.eye(REP_DIM), trainable=True, name="A")
transformation_A_bar = tf.Variable(tf.eye(REP_DIM), trainable=True, name="A_bar")

transformation_b = tf.Variable(tf.zeros([REP_DIM, 1]), trainable=True, name="b")

B_matrix = tf.Variable(tf.random.uniform([TRG_NUM_CLASSES, SRC_NUM_CLASSES], 0, 1).numpy(), trainable=True, name="B")

prior_D = tf.Variable(prior_source, trainable=True, name="D")
################

optimizer_A = tf.keras.optimizers.Adam(1E-2)
optimizer_A_bar = tf.keras.optimizers.Adam(1E-2)
optimizer_b = tf.keras.optimizers.Adam(1E-2)
optimizer_B = tf.keras.optimizers.Adam(1E-2)
optimizer_D = tf.keras.optimizers.Adam(1E-4)

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def get_gradient_norm(data, class_labels, random_labels):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        data = tf.convert_to_tensor(data, tf.float32)
        tape.watch(data)
        outputs = classifier(data, training=True)
        loss = ce_loss_none(class_labels, outputs) + ce_loss_none(random_labels, outputs)
     
    gradients_data_var = tape.gradient(loss, data)  
    g_norm2 = tf.sqrt(tf.reduce_sum(gradients_data_var ** 2, axis=[1]))
    max_grad_norm = tf.reduce_max(g_norm2)
    return max_grad_norm

def L2_dist(x, y):
    '''
    compute the squared L2 distance between two matrics
    '''
    dist_1 = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
    dist_2 = tf.reshape(tf.reduce_sum(tf.square(y), 1), [1, -1])
    dist_3 = 2.0 * tf.tensordot(x, tf.transpose(y), axes = 1) 
    return dist_1 + dist_2 - dist_3

@tf.function
def optimize_WD(src_reps, src_labels, trg_reps, trg_labels, wasserstein_mapping, cw_src_reps, cw_src_labels, num_pts):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        
        B_matrix_softmax = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1))
        prior_D_softmax = tf.nn.softmax(prior_D)
        
        new_src_reps = tf.transpose(tf.matmul(transformation_A, tf.transpose(src_reps)) + transformation_b) 
        new_src_labels = tf.matmul(src_labels, tf.transpose(B_matrix_softmax))
        
        new_trg_reps = trg_reps
        
        feature_cost_tf = L2_dist(new_src_reps, new_trg_reps)
        label_cost_tf = L2_dist(new_src_labels, trg_labels)
        
        WD_tf = tf.reduce_sum(tf.cast(wasserstein_mapping, tf.float32) * (feature_cost_tf + LAMBDA * label_cost_tf))
        
        cond_entropy = -tf.reduce_sum(prior_D_softmax * tf.reduce_sum(B_matrix_softmax * tf.math.log(B_matrix_softmax), axis=0))
        
        prior_matching = tf.reduce_sum(tf.square(prior_target - tf.reshape(tf.matmul(B_matrix_softmax, tf.reshape(prior_D_softmax, [-1,1])), [-1])))
        
        invertible_constraint = tf.reduce_sum(tf.square(tf.matmul(transformation_A, transformation_A_bar) - tf.eye(REP_DIM))) + \
                                tf.reduce_sum(tf.square(tf.matmul(transformation_A_bar, transformation_A) - tf.eye(REP_DIM)))
                            
        src_outputs = [classifier(cw_src_reps[idx], training=False) for idx in range(len(cw_src_reps))]
        weighted_src_loss = tf.cast(1/num_pts, tf.float32) * tf.reduce_sum([(prior_D_softmax[idx]/prior_source[idx]) * tf.reduce_sum(ce_loss_none(cw_src_labels[idx], src_outputs[idx]))for idx in range(len(cw_src_reps))])
        
        loss = (cond_entropy + weighted_src_loss) + LAMDA_WD * tf.reduce_sum(WD_tf) + LAMDA_PM * prior_matching + LAMDA_IC * invertible_constraint

    gradients_A = tape.gradient(loss, transformation_A)
    gradients_A_bar = tape.gradient(loss, transformation_A_bar)
    gradients_b = tape.gradient(loss, transformation_b)
    gradients_B = tape.gradient(loss, B_matrix)
    gradients_D = tape.gradient(loss, prior_D)
    
    optimizer_A.apply_gradients(zip([gradients_A], [transformation_A]))
    optimizer_A_bar.apply_gradients(zip([gradients_A_bar], [transformation_A_bar]))
    optimizer_b.apply_gradients(zip([gradients_b], [transformation_b])) 
    optimizer_B.apply_gradients(zip([gradients_B], [B_matrix])) 
    optimizer_D.apply_gradients(zip([gradients_D], [prior_D])) 
    
    return loss, tf.reduce_sum(WD_tf), cond_entropy, prior_matching, invertible_constraint, weighted_src_loss
   
for epoch in range(EPOCHS):
    
    b = transformation_b.numpy()
    A = transformation_A.numpy()
    A_bar = transformation_A_bar.numpy()
    B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
    D = tf.nn.softmax(prior_D).numpy()
    
    orig_source_train_rep, orig_source_y_train = get_set_based_on_prior(source_train_rep_all, source_y_train_all, BATCH_SIZE, prior_source, SRC_NUM_CLASSES)
    source_train_rep, source_y_train = get_set_based_on_prior(source_train_rep_all, source_y_train_all, BATCH_SIZE, D, SRC_NUM_CLASSES)
    target_train_rep, target_y_train = get_set_based_on_prior(target_train_rep_all, target_y_train_all, BATCH_SIZE, prior_target, TRG_NUM_CLASSES)
    
    modified_source_train_rep = (np.matmul(A, source_train_rep.T) + b).T
    modified_source_train_labels = tf.matmul(source_y_train, B.T).numpy()
    
    modified_target_train_rep = target_train_rep
    
    full_feature_cost = cdist(modified_source_train_rep, modified_target_train_rep, metric='euclidean')
    full_label_cost = cdist(modified_source_train_labels, target_y_train, metric='euclidean')
    
    full_cost = full_feature_cost + LAMBDA * full_label_cost
    
    full_wasserstein_mapping = ot.emd(ot.unif(modified_source_train_rep.shape[0]), ot.unif(target_train_rep.shape[0]), full_cost)
           
    min_src_pts = BATCH_SIZE
    for i in range(SRC_NUM_CLASSES):
        c_idx = np.argwhere(np.argmax(orig_source_y_train,1) == i).flatten()
        min_src_pts = min(min_src_pts, len(c_idx))
        
    cw_source_train_rep = []
    cw_source_y_train = []
    for i in range(SRC_NUM_CLASSES):
        c_idx = np.argwhere(np.argmax(orig_source_y_train,1) == i).flatten()
        np.random.shuffle(c_idx)
        c_idx = c_idx[:min_src_pts]
        cw_source_train_rep.append(tf.convert_to_tensor(orig_source_train_rep[c_idx], tf.float32))
        cw_source_y_train.append(tf.convert_to_tensor(orig_source_y_train[c_idx], tf.float32))
    
    l, wd, ent, pm, ic, ws = optimize_WD(tf.convert_to_tensor(source_train_rep, tf.float32), 
                                         tf.convert_to_tensor(source_y_train, tf.float32), 
                                         tf.convert_to_tensor(target_train_rep, tf.float32), 
                                         tf.convert_to_tensor(target_y_train, tf.float32), 
                                         tf.convert_to_tensor(full_wasserstein_mapping, tf.float32), 
                                         cw_source_train_rep, 
                                         cw_source_y_train,
                                         tf.convert_to_tensor(min_src_pts * SRC_NUM_CLASSES)
                                         )
    
    if epoch%100 == 0:
        
        b = transformation_b.numpy()
        A = transformation_A.numpy()
        A_bar = transformation_A_bar.numpy()
        B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
        D = tf.nn.softmax(prior_D).numpy()
        
        print(epoch, CHECKPOINT_PATH, len(source_train_rep_all),
              "WD:", np.round(np.sqrt(wd.numpy()),3),
              "CE:", np.round(ent.numpy(),3),
              "PM:", np.round(pm.numpy(),3),
              "IC:", np.round(ic.numpy(),3),
              "WS:", np.round(ws.numpy(),3),
              "ACC:", eval_accuracy_and_loss_cls_with_B(np.matmul(A_bar, target_test_rep_all.T - b).T, target_y_test_all, classifier, B, transpose_B = True)
              )
    
    
    if epoch!=0 and epoch%1000 == 0:    
        b = transformation_b.numpy()
        A = transformation_A.numpy()
        A_bar = transformation_A_bar.numpy()
        B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
        D = tf.nn.softmax(prior_D).numpy()
        
        CE = -tf.reduce_sum(D * tf.reduce_sum(B * tf.math.log(B), axis=0)).numpy()
        PM = tf.reduce_sum(tf.square(prior_target - np.matmul(B, D.reshape([-1,1])).reshape([-1]))).numpy()
        
        test_accs = eval_accuracy_and_loss_cls_with_B(np.matmul(A_bar, target_test_rep_all.T - b).T, target_y_test_all, classifier, B, transpose_B = True)
        
        modified_source_test_rep = (np.matmul(A, source_test_rep_all.T) + b).T
        modified_source_test_labels = tf.matmul(source_y_test_all, B.T).numpy()
        
        transformed_src_accs = eval_accuracy_and_loss_cls_with_B(np.matmul(A_bar, modified_source_test_rep.T - b).T, modified_source_test_labels, classifier, B, transpose_B = True)
       
        WD_test = compute_WD_same_label_sets(modified_source_test_rep, 
                                            modified_source_test_labels, 
                                            target_test_rep_all, 
                                            target_y_test_all,
                                            1) 
       
        weigted_SRC_loss = get_weigted_SRC_loss(source_test_rep_all, source_y_test_all, D/prior_source, SRC_NUM_CLASSES, classifier)
        
        TAU = get_gradient_norm(np.matmul(A_bar, target_test_rep_all.T - b).T, target_y_test_all, tf.keras.utils.to_categorical(np.random.randint(0, SRC_NUM_CLASSES, len(target_test_rep_all)), SRC_NUM_CLASSES)).numpy()
        
        print("SRC:", SRC, "TRG:", TRG, "Target test Accs: ", 
              np.round(test_accs[0], 3),
              np.round(test_accs[1], 3),
              np.round(transformed_src_accs[0], 3),
              np.round(transformed_src_accs[1], 3),
              np.round(weigted_SRC_loss, 3),
              np.round(WD_test, 3),
              np.round(CE, 3),
              np.round(weigted_SRC_loss + CE, 3),
              np.round(weigted_SRC_loss + CE + TAU * np.round(WD_test, 3), 3),
              np.round(transformed_src_accs[1] + TAU * np.round(WD_test, 3), 3),
              TAU
              )

print("################ END ################")
