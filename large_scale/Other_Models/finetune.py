import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
sys.path.append("../")
import sys
import clip
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import TransResNet
from utils import load_data, get_all_reps_pytorch, eval_accuracy_and_loss_cls_pytorch
import numpy as np
import torchvision.models as models
from sklearn.preprocessing import StandardScaler

print("################ START ################")
parser = argparse.ArgumentParser(description='Finetune from Imagenet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='imagenet', choices=['imagenet'], help='Name of SRC data')
parser.add_argument('--TRG', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'cifar100_small', 'cifar100_medium', 'pets', 'dtd', 'aircraft'], help='Name of TRG data')
parser.add_argument('--encoder', type=str, default='resnet50', choices=['resnet50'], help='Name of encoder')
parser.add_argument('--model', type=str, default='clip', choices=['simclr', 'moco', 'swav', 'robust', 'clip'], help='Name of model')
args = parser.parse_args()

SRC = args.SRC
TRG = args.TRG

TAU = 0.02

IS_CLIP = False
EPOCHS = 5001
BATCH_SIZE = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGENET = True if 'imagenet' in SRC else False
CHECKPOINT_PATH=f"../pretrained_models/{args.model}_{args.encoder}_1x.pth.tar"

###############
# Data
SRC_NUM_CLASSES = 1000
###############

###############
TRG_trainloader, TRG_testloader, TRG_NUM_CLASSES = load_data(TRG, BATCH_SIZE, IMAGENET, model=args.model)
###############


# Model
print('==> Building model..')
pretrained = models.__dict__[args.encoder]()
if args.model == 'simclr':
    state_dict = torch.load(CHECKPOINT_PATH)['state_dict']
    msg = pretrained.load_state_dict(state_dict)
    
elif args.model == 'moco':
    state = torch.load(CHECKPOINT_PATH)
    state_dict = state['state_dict']
    for k in list(state_dict.keys()):
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % 'fc'):
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            del state_dict[k]

    msg = pretrained.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % 'fc', "%s.bias" % 'fc'}

elif args.model == 'swav':
    pretrained = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    
elif args.model == 'clip':
    pretrained, preprocess = clip.load('ViT-B/32',device=torch.device("cpu"))
    IS_CLIP = True
else:
    print("Model not supported")
        
SRC_net = TransResNet(num_classes=SRC_NUM_CLASSES, IMAGENET=True, net=pretrained, clip=IS_CLIP)

SRC_net = SRC_net.to(DEVICE)

print('==> Freeze the encoder')
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

optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=5e-4)

relu = torch.nn.ReLU()
for epoch in range(EPOCHS):
    train_loss = 0
    correct = 0
    total = 0
    SRC_net.train()
    
    ind_batch = np.random.choice(np.arange(len(TRG_train_rep_all)), BATCH_SIZE)
    
    inputs = torch.tensor(TRG_train_rep_all[ind_batch]).to(DEVICE).requires_grad_(True)
    targets = torch.tensor(TRG_y_train_one_hot[ind_batch]).to(DEVICE)
    
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
        gradient_norm = torch.max(test_gradients.norm(2, dim=1))
        print("Max grad norm:", gradient_norm.item())
        print("\n")

print("################ END ################")
torch.cuda.empty_cache()
