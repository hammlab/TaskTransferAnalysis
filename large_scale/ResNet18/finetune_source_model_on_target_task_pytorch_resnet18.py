import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
sys.path.append("../")
import os, sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import TransResNet
from utils import load_data, get_all_reps_pytorch, eval_accuracy_and_loss_cls_pytorch
from torchvision.models import ResNet18_Weights, resnet18 
import numpy as np
import torchvision.models as models
from sklearn.preprocessing import StandardScaler

print("################ START ################")
parser = argparse.ArgumentParser(description='Finetune from Imagenet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='imagenet_small', choices=['imagenet_small'], help='Name of SRC data')
parser.add_argument('--TRG', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'cifar100_small', 'cifar100_medium', 'pets', 'dtd', 'aircraft'], help='Name of TRG data')
parser.add_argument('--ROBUST', type=bool, default=False, choices=[False,True], help='Use Robust Model or not')
parser.add_argument('--TAU', type=float, default=0.02, help='TAU')
parser.add_argument('--EPS', type=str, default=0.1, help='EPS')
args = parser.parse_args()
print(args)

SRC = args.SRC
TRG = args.TRG
ROBUST = bool(args.ROBUST)

TAU = float(args.TAU)
EPS = str(args.EPS)

REP_DIM = 512
EPOCHS = 5001
BATCH_SIZE = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGENET = True 

if ROBUST:
    CHECKPOINT_PATH = "./checkpoints/adversarially_trained/resnet18_l2_eps"+str(EPS)+".ckpt"
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit("No Model:"+CHECKPOINT_PATH)
else:
    CHECKPOINT_PATH="Pretrained Imagenet"

###############
# Data
SRC_NUM_CLASSES = 1000
###############

###############
TRG_trainloader, TRG_testloader, TRG_NUM_CLASSES = load_data(TRG, BATCH_SIZE, IMAGENET, transforming=True)
###############

assert SRC_NUM_CLASSES >= TRG_NUM_CLASSES

if ROBUST:
    pretrained = models.__dict__['resnet18']()
    state_dict = torch.load(CHECKPOINT_PATH)['model']
    for k in list(state_dict.keys()):
        if 'module.model.' in k and 'new' not in k:
            state_dict[k.replace('module.model.', '')] = state_dict[k]
        del state_dict[k]
    msg = pretrained.load_state_dict(state_dict)
    SRC_net = TransResNet(num_classes=SRC_NUM_CLASSES, IMAGENET = True, net=pretrained)
else:
    SRC_net = TransResNet(num_classes=SRC_NUM_CLASSES, IMAGENET = True, net=resnet18(weights = ResNet18_Weights.DEFAULT))

SRC_net = SRC_net.to(DEVICE)

if not IMAGENET:
    CKPT = torch.load(CHECKPOINT_PATH)
    SRC_net.load_state_dict(CKPT['net'])
    SRC_net.eval()
    print("Best acc loaded:", CKPT['acc'])
    
for name, param in SRC_net.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
        
num_ftrs = SRC_net.linear.in_features
SRC_net.linear = nn.Linear(num_ftrs, TRG_NUM_CLASSES)
SRC_net = SRC_net.to(DEVICE)

params_to_update = []
for name,param in SRC_net.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)        

TRG_train_rep_all, TRG_y_train_one_hot = get_all_reps_pytorch(TRG_trainloader, SRC_net, TRG_NUM_CLASSES, DEVICE, len(TRG_trainloader))
TRG_test_rep_all, TRG_y_test_one_hot = get_all_reps_pytorch(TRG_testloader, SRC_net, TRG_NUM_CLASSES, DEVICE, len(TRG_testloader))

trg_scaler = StandardScaler()
trg_scaler.fit(TRG_train_rep_all)

TRG_train_rep_all = trg_scaler.transform(TRG_train_rep_all)
TRG_test_rep_all = trg_scaler.transform(TRG_test_rep_all)

criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

if TRG == 'cifar10':
    optimizer = optim.SGD(params_to_update, lr=0.00001, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=5e-4)
    
relu = torch.nn.ReLU()
for epoch in range(EPOCHS):
    train_loss = 0
    correct = 0
    total = 0
    SRC_net.train()
    
    ind_batch = np.random.choice(np.arange(len(TRG_train_rep_all)), BATCH_SIZE)
    
    inputs = torch.tensor(TRG_train_rep_all[ind_batch]).to(DEVICE).requires_grad_(True)
    targets = torch.tensor(np.argmax(TRG_y_train_one_hot[ind_batch],1)).to(DEVICE)
    target_random = torch.tensor(np.random.randint(0, TRG_NUM_CLASSES, BATCH_SIZE)).to(DEVICE)

    optimizer.zero_grad()
    
    preds = SRC_net.classify(inputs)
    
    loss_ce = criterion(preds, targets)
    loss_ce_none = criterion_none(preds, targets)
    loss_ce_none_random = criterion_none(preds, target_random)
    
    grad_outputs = torch.ones(loss_ce_none.size(), device=DEVICE, requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=loss_ce_none,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    
    gradient_penalty = torch.mean(relu(gradients.norm(2, dim=1) - TAU)**2)
    
    gradients_random = torch.autograd.grad(
        outputs=loss_ce_none_random,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients_random = gradients_random.view(gradients_random.size(0), -1)
    
    gradient_random_penalty = torch.mean(relu(gradients_random.norm(2, dim=1) - TAU)**2)
    
    loss = loss_ce + 1E4 * (gradient_penalty + gradient_random_penalty)
    
    loss.backward()
    optimizer.step()
        
        
    if epoch % 1000 == 0:
        test_accuracy = eval_accuracy_and_loss_cls_pytorch(TRG_test_rep_all, TRG_y_test_one_hot, SRC_net, DEVICE)
        print("Epoch:", epoch)
        print("Record SRC:", SRC, "Test:", test_accuracy[0], test_accuracy[1], "Grad penalty:", gradient_penalty.item())
        
        target_test_random = torch.tensor(np.random.randint(0, TRG_NUM_CLASSES, len(TRG_test_rep_all))).to(DEVICE)
        test_inputs = torch.tensor(TRG_test_rep_all).to(DEVICE).requires_grad_(True)
        test_preds = SRC_net.classify(test_inputs)
        test_loss_ce_none_random = criterion_none(test_preds, target_test_random)
        
        test_grad_outputs = torch.ones(test_loss_ce_none_random.size(), device=DEVICE, requires_grad=False)
        
        test_gradients = torch.autograd.grad(
            outputs=test_loss_ce_none_random,
            inputs=test_inputs,
            grad_outputs=test_grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        test_gradients = test_gradients.view(test_gradients.size(0), -1)
        gradient_norm = torch.max(gradients.norm(2, dim=1))
        print("Max grad norm:", gradient_norm.item())
        print("\n")

print("################ END ################")