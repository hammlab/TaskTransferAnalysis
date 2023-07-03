import matplotlib
matplotlib.use('Agg')
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import ot
from scipy.spatial.distance import cdist 
from sklearn.manifold import TSNE
import gzip
import pickle as cPickle
from torchvision import datasets
ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
ce_loss_none_not_from_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)


def load_data(dataset):
    
    if dataset == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'FMNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'USPS':
        x_train, y_train, x_test, y_test = load_usps(all_use=False)

        x_train = 255.*x_train
        x_test = 255.*x_test
        
    elif dataset == 'KMNIST':
        x_train, y_train, x_test, y_test = load_kmnist()
    
    elif dataset == 'EMNIST':
        train_set = datasets.EMNIST(root="Datasets/data", split="letters", train=True,download=True)
        test_set = datasets.EMNIST(root="Datasets/data", split="letters", train=False,download=True)
        
        x_train, y_train = train_set.data.numpy(), train_set.targets.numpy()
        x_test, y_test = test_set.data.numpy(), test_set.targets.numpy()
        y_train = y_train - 1
        y_test = y_test - 1
        
    else:
        sys.exit("unsupported dataset")
    
    return x_train, y_train, x_test, y_test

def load_usps(all_use=True):
    
    root = "Datasets/data"
    
    f = gzip.open(root+'/usps_28x28.pkl', 'rb')
    data_set = cPickle.load(f, encoding='latin1')
    f.close()
    img_train = data_set[0][0]
    label_train = data_set[0][1]
    img_test = data_set[1][0]
    label_test = data_set[1][1]
    inds = np.random.permutation(img_train.shape[0])
    if all_use:
        img_train = img_train[inds][:6562]
        label_train = label_train[inds][:6562]
    else:
        img_train = img_train[inds][:1800]
        label_train = label_train[inds][:1800]
    img_train = img_train.reshape((img_train.shape[0], 28, 28, 1))
    img_test = img_test.reshape((img_test.shape[0], 28, 28, 1))
    return img_train, label_train, img_test, label_test

def load_kmnist():
    def load(f):
        return np.load(f)['arr_0']

    root = 'Datasets/data/'
    x_train = load(root+'kmnist-train-imgs.npz')
    x_test = load(root+'kmnist-test-imgs.npz')
    y_train = load(root+'kmnist-train-labels.npz')
    y_test = load(root+'kmnist-test-labels.npz')
    
    return x_train, y_train, x_test, y_test

def compute_WD_same_label_sets(data_S, labels_S, data_T, labels_T, std_orig_S):
    
    failure = 0
    data_S = data_S/std_orig_S
    data_T = data_T/std_orig_S

    wd_num = 0
    NUM_CLASSES = len(np.unique(np.argmax(labels_S, 1)))
    
    for k in range(NUM_CLASSES):
        idx_S = np.argwhere(np.argmax(labels_S, 1) == k).flatten()
        idx_T = np.argwhere(np.argmax(labels_T, 1) == k).flatten()
        if (len(idx_S) == 0 and len(idx_T) != 0) or (len(idx_S) != 0 and len(idx_T) == 0):
            print("Class ", k, "has zero samples in S or T", len(idx_S), len(idx_T))
            print("Aborting WD computation")
            failure=1
            break        
        
        C = cdist(data_S[idx_S], data_T[idx_T], metric='euclidean')
            
        gamma = ot.emd(ot.unif(len(idx_S)), ot.unif(len(idx_T)), C)
        wd_num += np.sum(gamma * C) / NUM_CLASSES
    
    if failure==1:
        WDs = 1E10
    else:
        WDs = wd_num
    
    return WDs

def get_all_reps(data_x, shared):
    batch_size = 200
    nb_batches = int(len(data_x)/batch_size)
    if len(data_x)%batch_size != 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(data_x)))
        sha = shared(data_x[ind_batch], training=False).numpy()
        if batch == 0:
            all_reps = sha
        else:
            all_reps = np.concatenate([all_reps, sha])
            
    return all_reps

def get_weigted_SRC_loss(source_test_rep_all, source_y_test_all, C, SRC_NUM_CLASSES, classifier):
    batch_size = 200
    weigted_loss = 0
    for i in range(SRC_NUM_CLASSES):
        c_idx = np.argwhere(np.argmax(source_y_test_all, 1) == i).flatten()
    
        nb_batches = int(len(c_idx)/batch_size)
        if len(c_idx)%batch_size != 0:
            nb_batches += 1
           
        cls_loss = 0
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(c_idx)))
            cls_outputs = classifier(source_test_rep_all[c_idx[ind_batch]], training=False)
            cls_loss += np.sum(ce_loss_none(source_y_test_all[c_idx[ind_batch]], cls_outputs).numpy())
        
        weigted_loss += C[i] * cls_loss
        
    weigted_loss /= len(source_y_test_all)
    return weigted_loss
    

def eval_accuracy(x_test, y_test, shared, classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        sha = shared(x_test[ind_batch], training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
    
    acc /= np.float32(len(x_test))
    return acc*100

def eval_accuracy_and_loss(x_test, y_test, shared, classifier):
    acc = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        sha = shared(x_test[ind_batch], training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def eval_accuracy_cls(x_test, y_test, classifier):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
    
    acc /= np.float32(len(x_test))
    return acc*100

def eval_accuracy_and_loss_cls(x_test, y_test, classifier):
    acc = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def eval_accuracy_cls_with_B(x_test, y_test, classifier, B_matrix, transpose_B = False):
    acc = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        if transpose_B:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix.T)
        else:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
    
    acc /= np.float32(len(x_test))
    return acc*100

def eval_accuracy_and_loss_cls_with_B(x_test, y_test, classifier, B_matrix, transpose_B = False):
    
    acc = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        if transpose_B:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix.T)
        else:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix)
                
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        loss += np.sum(ce_loss_none_not_from_logits(y_test[ind_batch], pred).numpy())
    
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def eval_accuracy_and_loss_cls_with_B_and_C(x_test, y_test, classifier, B_matrix, C_mul, transpose_B = False):
    acc = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        pred = classifier(x_test[ind_batch], training=False)
        pred = tf.nn.softmax(pred).numpy()
        pred = C_mul * pred 
        pred = pred/np.sum(pred)
        if transpose_B:
            pred = np.matmul(pred, B_matrix.T)
        else:
            pred = np.matmul(pred, B_matrix)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        loss += np.sum(ce_loss_none_not_from_logits(y_test[ind_batch], pred).numpy())
    
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def get_balanced_set(X, Y, points, num_classes, shuffle = True):
    assert Y.shape[1] == num_classes
    classes = np.unique(np.argmax(Y, 1))
    num_per_class = int(points / len(classes))
    for i in range(len(classes)):
        clss = np.argwhere(np.argmax(Y, 1) == classes[i]).flatten()
        np.random.shuffle(clss)
        clss = clss[:num_per_class]
        if i == 0:
            X_ = np.array(X[clss])
            Y_ = np.array(Y[clss])
        else:
            X_ = np.concatenate([X_, X[clss]])
            Y_ = np.concatenate([Y_, Y[clss]])
            
    if shuffle:
        idx = np.arange(len(X_))
        np.random.shuffle(idx)
        X_ = X_[idx]
        Y_ = Y_[idx]
    return X_, Y_

def get_set_based_on_prior(X, Y, TOTAL_POINTS, prior, num_classes, shuffle = True):
    assert Y.shape[1] == num_classes
    classes = np.unique(np.argmax(Y, 1))
    for i in range(len(classes)):
        clss = np.argwhere(np.argmax(Y, 1) == classes[i]).flatten()
        np.random.shuffle(clss)
        
        points_per_class = min(len(clss), int(TOTAL_POINTS*prior[i]))
        
        clss = clss[:points_per_class]
        if i == 0:
            X_ = np.array(X[clss])
            Y_ = np.array(Y[clss])
        else:
            X_ = np.concatenate([X_, X[clss]])
            Y_ = np.concatenate([Y_, Y[clss]])
            
    if shuffle:
        idx = np.arange(len(X_))
        np.random.shuffle(idx)
        X_ = np.array(X_[idx])
        Y_ = np.array(Y_[idx])
    return X_, Y_

def plot_embedding(X, y, d, TYPE, transform=False):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0)-2, np.max(X, 0)+2
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    
    for i in range(X.shape[0]):
        
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.tab10(d[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    
    plt.xticks([]), plt.yticks([])
    plt.title("")
    
    s_patch = mpatches.Patch(color=plt.cm.tab10(0./10.), label='Source data')
    t_patch = mpatches.Patch(color=plt.cm.tab10(1./10.), label='Target data')
    if transform:
        tr_patch = mpatches.Patch(color=plt.cm.tab10(2./10.), label='Transformed data')
        plt.legend(handles=[s_patch, t_patch, tr_patch], loc='best')
    else:
        plt.legend(handles=[s_patch, t_patch], loc='best')
    
    plt.savefig('Plots/'+str(TYPE)+'.pdf', bbox_inches='tight')
        
    plt.close()
    
    
def plot_images(images, rows, cols, method):
    
    fig = plt.figure(1,(128, 10.))
    grid = ImageGrid(fig, 121, nrows_ncols=(rows, cols), axes_pad = 0.01)
    for i in range(rows*cols):
        grid[i].imshow(images[i])
        grid[i].axis('off')
    plt.savefig("Plots/"+str(method)+".pdf", bbox_inches='tight')
    plt.close()
    
def plot_tsne(A, b, source_rep, source_y, target_rep, target_y, TOTAL_SAMPLES, SRC_NUM_CLASSES, TRG_NUM_CLASSES, filename):
    
    source_tsne_images, source_tsne_labels = get_balanced_set(source_rep, source_y, TOTAL_SAMPLES, SRC_NUM_CLASSES)
    target_tsne_images, target_tsne_labels = get_balanced_set(target_rep, target_y, TOTAL_SAMPLES, TRG_NUM_CLASSES)
    
    tsne = TSNE(n_components=2, n_iter=3000, n_jobs=2, random_state=0)
    plot_tsne = tsne.fit_transform(np.concatenate([source_tsne_images, 
                                                   (np.matmul(A, target_tsne_images.T)+b).T
                                                   ]), 
                                   np.concatenate([np.argmax(source_tsne_labels, 1), 
                                                   np.argmax(target_tsne_labels, 1),
                                                   ]))
    
    plot_embedding(plot_tsne, 
                   np.concatenate([np.argmax(source_tsne_labels, 1), 
                                   np.argmax(target_tsne_labels, 1),
                                   ]), 
                   np.concatenate([np.zeros(len(source_tsne_labels)), 
                                   2*np.ones(len(target_tsne_labels))
                                   ]), 
                   filename, True)
    
