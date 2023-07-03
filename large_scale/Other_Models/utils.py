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
from torchvision import datasets
from scipy.io import loadmat
import torchvision.transforms as transforms
import torch
import os
import time
import socket
sys.path.append("../robust-models-transfer/src/robustness_main/")
sys.path.append("../robust-models-transfer/src/")
import torch as ch
from robustness import datasets as robustness_datasets
from utils_robustness import transfer_datasets
from torch.utils.data import Subset
import torch.nn as nn
import clip
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_transformation_dict = {
    'robust':  transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]),
    'simclr': transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]),
    'moco': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
    'swav': transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]),
            
}

test_transformation_dict = {
    'robust': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ]),
    'simclr': transforms.Compose([
                    transforms.Resize(224, interpolation=Image.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]),
    'moco': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),
    'swav': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]),   
}    

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
ce_loss_none_not_from_logits = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

def get_dataset_and_loaders(dataset, BATCH_SIZE):
    ds, (train_loader, test_loader) = transfer_datasets.make_loaders(dataset, batch_size = BATCH_SIZE, workers = 8, subset = None)
    if type(ds) == int:
        new_ds = robustness_datasets.CIFAR("/tmp")
        new_ds.num_classes = ds
        new_ds.mean = ch.tensor([0., 0., 0.])
        new_ds.std = ch.tensor([1., 1., 1.])
        ds = new_ds
    return train_loader, test_loader

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense),))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i] = t
        else:
            labels_one_hot[i] = t
    return labels_one_hot


def load_data(dataset, BATCH_SIZE, IMAGENET, NUM_CLS=None, model='simclr'):
    
    root = "Datasets/data"
     
    small_train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    small_test_transforms = [
        transforms.ToTensor(),
    ]
    
    if IMAGENET:
        if model == 'clip':
            _, preprocess = clip.load('ViT-B/32', device='cpu')
            test_transformation_dict[model] = preprocess
            
        transform_train = test_transformation_dict[model]
        transform_test = test_transformation_dict[model]
    else:
        transform_train = transforms.Compose(small_train_transforms)
        transform_test = transforms.Compose(small_test_transforms)
    
    if dataset == 'cifar10':
        train_set = datasets.CIFAR10(root=root+"/data", train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR10(root=root+"/data", train=False, transform=transform_test, download=True)
        
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        NUM_CLASSES = 10
        
    elif dataset == 'cifar100':
        train_set = datasets.CIFAR100(root=root+"/data", train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR100(root=root+"/data", train=False, transform=transform_test, download=True)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

        NUM_CLASSES = 100
    
    elif dataset == 'cifar100_medium':
        NUM_CLASSES = 50
        
        train_set = datasets.CIFAR100(root=root+"/data", train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR100(root=root+"/data", train=False, transform=transform_test, download=True)
        
        train_idx = []
        test_idx = []
        for i in range(NUM_CLASSES):
            train_idx.append(torch.tensor(train_set.targets) == i)
            test_idx.append(torch.tensor(test_set.targets) == i)
        
        for i in range(NUM_CLASSES):
            if i ==0:
                train_mask = train_idx[0]
                test_mask = test_idx[0]
            else:
                train_mask = train_mask | train_idx[i]
                test_mask = test_mask | test_idx[i] 
               
        train_indices = train_mask.nonzero().reshape(-1)
        test_indices = test_mask.nonzero().reshape(-1)
        print(len(train_indices), len(test_indices))
        
        train_subset = Subset(train_set, train_indices)
        test_subset = Subset(test_set, test_indices)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
     
    elif dataset == 'cifar100_small':
        NUM_CLASSES = 25
        
        train_set = datasets.CIFAR100(root=root+"/data", train=True, transform=transform_train, download=True)
        test_set = datasets.CIFAR100(root=root+"/data", train=False, transform=transform_test, download=True)
        
        train_idx = []
        test_idx = []
        for i in range(NUM_CLASSES):
            train_idx.append(torch.tensor(train_set.targets) == i)
            test_idx.append(torch.tensor(test_set.targets) == i)
        
        for i in range(NUM_CLASSES):
            if i ==0:
                train_mask = train_idx[0]
                test_mask = test_idx[0]
            else:
                train_mask = train_mask | train_idx[i]
                test_mask = test_mask | test_idx[i] 
               
        train_indices = train_mask.nonzero().reshape(-1)
        test_indices = test_mask.nonzero().reshape(-1)
        
        train_subset = Subset(train_set, train_indices)
        test_subset = Subset(test_set, test_indices)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        
    elif dataset == 'imagenet_small':
        NUM_CLASSES = NUM_CLS
        
        all_classes = np.arange(1000)
        np.random.shuffle(all_classes)
        selected_classes = np.array(sorted(all_classes[:NUM_CLASSES]))
        not_selected_classes = all_classes[NUM_CLASSES:]
        
        train_set = datasets.ImageNet(root=root+"/data/Imagenet/train", split= 'train', transform=transform_train)
        test_set = datasets.ImageNet(root=root+"/data/Imagenet", split= 'val', transform=transform_test)
        
        train_set.targets = torch.tensor(train_set.targets)
        test_set.targets = torch.tensor(test_set.targets)
        
        for i in range(len(not_selected_classes)):
            train_set.targets = torch.where(train_set.targets == not_selected_classes[i], 1001, train_set.targets)
            test_set.targets = torch.where(test_set.targets == not_selected_classes[i], 1001, test_set.targets)
        
        for i, lab in enumerate(selected_classes):
            train_set.targets = torch.where(train_set.targets == lab, i, train_set.targets)
            test_set.targets = torch.where(test_set.targets == lab, i, test_set.targets)
        
        train_idx = []
        test_idx = []
        for i in range(NUM_CLASSES):
            train_idx.append(train_set.targets == i)
            test_idx.append(test_set.targets == i)
        
        for i in range(NUM_CLASSES):
            if i ==0:
                train_mask = train_idx[0]
                test_mask = test_idx[0]
            else:
                train_mask = train_mask | train_idx[i]
                test_mask = test_mask | test_idx[i] 
               
        train_indices = train_mask.nonzero().reshape(-1)
        test_indices = test_mask.nonzero().reshape(-1)
        
        train_subset = Subset(train_set, train_indices)
        test_subset = Subset(test_set, test_indices)
        
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
        
        
    elif dataset == 'aircraft' :
        trainloader, testloader = get_dataset_and_loaders(dataset, BATCH_SIZE)
        NUM_CLASSES = 100
        
    elif dataset == 'pets' :
        trainloader, testloader = get_dataset_and_loaders(dataset, BATCH_SIZE)
        NUM_CLASSES = 37
        
    elif dataset == 'dtd' :
        trainloader, testloader = get_dataset_and_loaders(dataset, BATCH_SIZE)
        NUM_CLASSES = 47
        
    else:
        sys.exit("unsupported dataset:"+dataset)
    
    if dataset in ['imagenet_small']:
        return trainloader, testloader, NUM_CLASSES, selected_classes
    return trainloader, testloader, NUM_CLASSES
        

def compute_WD_same_label_sets(data_S, labels_S, data_T, labels_T, std_orig_S):
    
    failure = 0
    data_S = data_S / std_orig_S
    data_T = data_T / std_orig_S

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

def get_all_reps_pytorch(loader, net, NUM_CLASSES, DEVICE, NUM_BATCHES, SELECTED_CLASSES=None):
    
    net.eval()
    
    if SELECTED_CLASSES is not None:
        label_dict = {}
        for i in range(len(SELECTED_CLASSES)):
            label_dict[SELECTED_CLASSES[i]]=i
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if SELECTED_CLASSES is not None:
                targets = torch.tensor(np.array([label_dict[j] for j in targets.cpu().numpy()])).to(DEVICE)
            
            _, reps = net(inputs)
            reps = reps.cpu().numpy()
            if batch_idx == 0:
                all_reps = reps
                all_trgs = tf.keras.utils.to_categorical(targets.cpu().numpy(), NUM_CLASSES)
            else:
                all_reps = np.concatenate([all_reps, reps])
                all_trgs = np.concatenate([all_trgs, tf.keras.utils.to_categorical(targets.cpu().numpy(), NUM_CLASSES)])
                
            if batch_idx == NUM_BATCHES-1:
                break
    
    return all_reps, all_trgs

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

def get_weigted_SRC_loss_pytorch(source_test_rep_all, source_y_test_all_one_hot, C, SRC_NUM_CLASSES, classifier, DEVICE):
    classifier.eval()
    batch_size = 200
    weigted_loss = 0
    for i in range(SRC_NUM_CLASSES):
        c_idx = np.argwhere(np.argmax(source_y_test_all_one_hot, 1) == i).flatten()
    
        nb_batches = int(len(c_idx)/batch_size)
        if len(c_idx)%batch_size != 0:
            nb_batches += 1
           
        cls_loss = 0
        for batch in range(nb_batches):
            ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(c_idx)))
            
            cls_outputs = classifier.classify(torch.tensor(source_test_rep_all[c_idx[ind_batch]]).to(DEVICE)).cpu().detach().numpy()
            
            cls_loss += np.sum(ce_loss_none(source_y_test_all_one_hot[c_idx[ind_batch]], cls_outputs).numpy())
            
        weigted_loss += C[i] * cls_loss
        
    weigted_loss /= len(source_y_test_all_one_hot)
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

def eval_accuracy_and_loss(x_test, y_test, base_model, encoder, classifier):
    acc = 0
    loss = 0
    batch_size = 200
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        sha = encoder(base_model(x_test[ind_batch], training=False), training=False)
        pred = classifier(sha, training=False)
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test[ind_batch],1))
        loss += np.sum(ce_loss_none(y_test[ind_batch], pred).numpy())
    
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def eval_accuracy_and_loss_pytorch(test_loader, net, NUM_CLASSES, DEVICE, class_weights=None):
    net.eval()
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            outputs, _ = net(inputs)

            loss_vec = ce_loss_none(tf.one_hot(targets.cpu().numpy(), NUM_CLASSES), outputs.cpu().numpy()).numpy()
            _, predicted = outputs.max(1)
            correct_vec = predicted.eq(targets)
            if class_weights is not None:
                test_loss += np.sum(loss_vec * class_weights[targets.cpu()])
                correct += (correct_vec.cpu() * class_weights[targets.cpu()]).sum().item()
            else:
                test_loss += np.sum(loss_vec)
                correct += correct_vec.sum().item()
            
            total += targets.size(0)
            
    acc = 100.*correct/total
    test_loss /= total
    return acc, test_loss

def eval_accuracy_and_loss_check(x_test, y_test, shared, classifier):
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

def eval_accuracy_and_loss_cls_pytorch(x_test, y_test_one_hot, classifier, DEVICE):
    
    classifier.eval()
    acc = 0
    loss = 0
    batch_size = 200
    points = 0
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        pred = classifier.classify(torch.tensor(x_test[ind_batch]).to(DEVICE))
        
        pred = pred.cpu().detach().numpy()
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test_one_hot[ind_batch],1))
        loss += np.sum(ce_loss_none(y_test_one_hot[ind_batch], pred).numpy())
        
        points += len(ind_batch)
    
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

def eval_accuracy_and_loss_cls_with_B_pytorch(x_test, y_test_one_hot, classifier, B_matrix, DEVICE, transpose_B = False):
    classifier.eval()
    acc = 0
    loss = 0
    batch_size = 200
    points = 0
    nb_batches = int(len(x_test)/batch_size)
    if len(x_test)%batch_size!= 0:
        nb_batches += 1
    for batch in range(nb_batches):
        ind_batch = range(batch_size*batch, min(batch_size*(1+batch), len(x_test)))
        
        pred = classifier.classify(torch.tensor(x_test[ind_batch]).to(DEVICE)).cpu().detach().numpy()
        
        if transpose_B:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix.T)
        else:
            pred = np.matmul(tf.nn.softmax(pred).numpy(), B_matrix)
        
        acc += np.sum(np.argmax(pred,1) == np.argmax(y_test_one_hot[ind_batch],1))
        loss += np.sum(ce_loss_none_not_from_logits(y_test_one_hot[ind_batch], pred).numpy())
    
        points += len(ind_batch)
    loss /= np.float32(len(x_test))
    acc /= np.float32(len(x_test))
    return acc*100, loss

def get_grad_norm_pytorch(x_src, y_src, x_trg, y_trg, classifier, B_matrix, DEVICE, TRG_NUM_CLASSES):
    classifier.eval()
    criterion=nn.NLLLoss(reduction='none')
    
    inputs = torch.tensor(np.concatenate([x_src, x_trg, x_trg])).to(DEVICE).requires_grad_(True)
    y = torch.tensor(np.concatenate([np.random.randint(0, TRG_NUM_CLASSES, len(y_src)), np.random.randint(0, TRG_NUM_CLASSES, len(y_trg)), np.argmax(y_trg, 1)])).to(DEVICE)    
    
    outputs = torch.nn.functional.softmax(classifier.classify(inputs), dim=-1)
    preds = torch.log(torch.matmul(outputs, torch.transpose(torch.tensor(B_matrix).to(DEVICE), 0, 1)))
    loss = criterion(preds, y)
    
    grad_outputs = torch.ones(loss.size(), device=DEVICE, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = torch.norm(gradients, p=2, dim=1)
    print("Min:", torch.min(gradient_norm).item(), "Max:", torch.max(gradient_norm).item(), "Mean:", torch.mean(gradient_norm).item())
    
    return torch.max(gradient_norm).item()

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
    for i in range(num_classes):
        clss = np.argwhere(np.argmax(Y, 1) == i).flatten()
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

def get_set_based_on_prior_with_clss(X, Y, TOTAL_POINTS, prior, num_classes, selected_clss, shuffle = True):
    assert Y.shape[1] == num_classes
    for i in range(num_classes):
        if shuffle:
            np.random.shuffle(selected_clss[i])
        
        points_per_class = min(len(selected_clss[i]), int(TOTAL_POINTS*prior[i]))
        
        clss = selected_clss[i][:points_per_class]
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

