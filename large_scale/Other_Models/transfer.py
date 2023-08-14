import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys
sys.path.append(".././/")
import sys
import numpy as np
from utils import compute_WD_same_label_sets, load_data, get_set_based_on_prior_with_clss
import ot
import tensorflow as tf
import argparse
import torch
from models import TransResNet
from utils import get_all_reps_pytorch, eval_accuracy_and_loss_cls_pytorch, eval_accuracy_and_loss_cls_with_B_pytorch, get_grad_norm_pytorch
import torchvision.models as models
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import clip

print("################ START ################")
parser = argparse.ArgumentParser(description='Finetune', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--SRC', type=str, default='imagenet_small', choices=['imagenet_small', 'cifar100_selected'], help='Name of SRC data')
parser.add_argument('--TRG', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'cifar100_small', 'cifar100_medium', 'pets', 'dtd', 'aircraft'], help='Name of TRG data')
parser.add_argument('--TAU', type=float, default=0.02, help='TAU')
parser.add_argument('--encoder', type=str, default='resnet50', choices=['resnet50'], help='Name of encoder')
parser.add_argument('--model', type=str, default='clip', choices=['simclr', 'moco', 'swav', 'clip'], help='Name of model')
args = parser.parse_args()
print(args)

SRC = args.SRC
TRG = args.TRG

IS_CLIP = False
if args.model == 'clip':
    REP_DIM = 512
    BATCH_SIZE = 1500
else:
    REP_DIM = 2048
    BATCH_SIZE = 2500
    
EPOCHS = 2001

LAMDA_WD = 2
LAMDA_PM = 1
LAMDA_IC = 1

TAU = float(args.TAU)
LAMBDA = 1E6
IMAGENET = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE_finetuning = 1000

CHECKPOINT_PATH=f"../pretrained_models/{args.model}_{args.encoder}_1x.pth.tar"

        
###############
TRG_trainloader, TRG_testloader, TRG_NUM_CLASSES = load_data(TRG, 200, IMAGENET, model=args.model)

if TRG in ['cifar100_small', 'cifar100_medium', 'dtd', 'aircraft']:
    prior_target = np.zeros(TRG_NUM_CLASSES, np.float32)
    total_samples = 0
    for batch_idx, (inputs, targets) in enumerate(TRG_trainloader):
        for i in range(TRG_NUM_CLASSES):
            prior_target[i] += len(np.argwhere(targets==i).flatten())
        total_samples += len(targets)
    prior_target /= np.float32(total_samples)
else:
    trg_targets = TRG_trainloader.dataset.targets
    counts = np.unique(trg_targets, return_counts=True)[1]
    prior_target = np.array(counts/len(trg_targets), np.float32)
###############

###############
# Data
if SRC in ['imagenet_small']:
    SRC_trainloader, SRC_testloader, SRC_NUM_CLASSES, SELECTED_CLASSES = load_data(SRC, 200, IMAGENET, TRG_NUM_CLASSES, model=args.model)
else:
    SRC_trainloader, SRC_testloader, SRC_NUM_CLASSES = load_data(SRC, 200, IMAGENET, model=args.model)
###############

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

print('==> Getting reps from the model for SRC..')
SRC_train_rep_all, SRC_y_train_one_hot = get_all_reps_pytorch(SRC_trainloader, SRC_net, SRC_NUM_CLASSES, DEVICE, 50, SELECTED_CLASSES)
SRC_test_rep_all, SRC_y_test_one_hot = get_all_reps_pytorch(SRC_testloader, SRC_net, SRC_NUM_CLASSES, DEVICE, len(SRC_testloader), SELECTED_CLASSES)

TRG_train_rep_all, TRG_y_train_one_hot = get_all_reps_pytorch(TRG_trainloader, SRC_net, TRG_NUM_CLASSES, DEVICE, len(TRG_trainloader))
TRG_test_rep_all, TRG_y_test_one_hot = get_all_reps_pytorch(TRG_testloader, SRC_net, TRG_NUM_CLASSES, DEVICE, len(TRG_testloader))

src_scaler = StandardScaler()
trg_scaler = StandardScaler()

src_scaler.fit(SRC_train_rep_all)
trg_scaler.fit(TRG_train_rep_all)

SRC_train_rep_all = src_scaler.transform(SRC_train_rep_all)
SRC_test_rep_all = src_scaler.transform(SRC_test_rep_all)

TRG_train_rep_all = trg_scaler.transform(TRG_train_rep_all)
TRG_test_rep_all = trg_scaler.transform(TRG_test_rep_all)

prior_source = np.zeros(SRC_NUM_CLASSES, np.float32)
for i in range(SRC_NUM_CLASSES):
    prior_source[i] = len(np.argwhere(np.argmax(SRC_y_train_one_hot, 1)==i).flatten())
prior_source /= np.float32(len(SRC_y_train_one_hot))
print(prior_source)

####################  Training h_S with gradient penalty ###################
print('==> Freeze the encoder')
for name, param in SRC_net.named_parameters():
    if 'encoder' in name:
        param.requires_grad = False
        
num_ftrs = SRC_net.linear.in_features
SRC_net.linear = nn.Linear(num_ftrs, SRC_NUM_CLASSES)
SRC_net = SRC_net.to(DEVICE)

params_to_update = []
for name,param in SRC_net.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)        

criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

if TRG == 'cifar10':
    optimizer = optim.SGD(params_to_update, lr=0.00001, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(params_to_update, lr=0.0001, momentum=0.9, weight_decay=5e-4)

print('==> Finetuning the SRC model..', print(len(SRC_trainloader)))

relu = torch.nn.ReLU()
for epoch in range(2001):
    train_loss = 0
    correct = 0
    total = 0
    SRC_net.train()
    
    ind_batch = np.random.choice(np.arange(len(SRC_train_rep_all)), BATCH_SIZE_finetuning)
    
    inputs = torch.tensor(SRC_train_rep_all[ind_batch]).to(DEVICE).requires_grad_(True)
    targets = torch.tensor(np.argmax(SRC_y_train_one_hot[ind_batch], 1)).to(DEVICE)
    target_random = torch.tensor(np.random.randint(0, SRC_NUM_CLASSES, BATCH_SIZE_finetuning)).to(DEVICE)

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
        print(args)
        print("Path:", CHECKPOINT_PATH, "IMAGENET:", IMAGENET)
        test_accuracy = eval_accuracy_and_loss_cls_pytorch(SRC_test_rep_all, SRC_y_test_one_hot, SRC_net, DEVICE)
        print("Epoch:", epoch)
        print("Record SRC:", SRC, "Test:", test_accuracy[0], test_accuracy[1], "Grad penalty:", gradient_penalty.item(), gradient_random_penalty.item())
        
        target_test_random = torch.tensor(np.random.randint(0, SRC_NUM_CLASSES, len(SRC_test_rep_all))).to(DEVICE)
        test_inputs = torch.tensor(SRC_test_rep_all).to(DEVICE).requires_grad_(True)
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
optimizer_D = tf.keras.optimizers.Adam(1E-3)    

ce_loss_none = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def L2_dist(x, y):
    '''
    compute the squared L2 distance between two matrics
    '''
    dist_1 = tf.reshape(tf.reduce_sum(tf.square(x), 1), [-1, 1])
    dist_2 = tf.reshape(tf.reduce_sum(tf.square(y), 1), [1, -1])
    dist_3 = 2.0 * tf.tensordot(x, tf.transpose(y), axes = 1) 
    return dist_1 + dist_2 - dist_3

@tf.function
def optimize_WD(src_reps, src_labels, trg_reps, trg_labels, wasserstein_mapping, cw_src_loss, num_pts):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    
    with tf.GradientTape(persistent=True) as tape:
        
        B_matrix_softmax = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1))
        
        prior_D_softmax = tf.nn.softmax(prior_D)
        
        new_src_reps = tf.transpose(tf.matmul(transformation_A, tf.transpose(src_reps)) + transformation_b) 
        new_src_labels = tf.matmul(src_labels, tf.transpose(B_matrix_softmax))
        
        feature_cost_tf = L2_dist(new_src_reps, trg_reps)
        label_cost_tf = L2_dist(new_src_labels, trg_labels)
        
        WD_tf = tf.reduce_sum(tf.cast(wasserstein_mapping, tf.float32) * (feature_cost_tf + LAMBDA * label_cost_tf))
        
        cond_entropy = -tf.reduce_sum(tf.math.multiply(prior_D_softmax, tf.reduce_sum(B_matrix_softmax * tf.math.log(B_matrix_softmax), axis=0)))
        
        prior_matching = tf.reduce_sum(tf.square(prior_target - tf.reshape(tf.matmul(B_matrix_softmax, tf.reshape(prior_D_softmax, [-1,1])), [-1])))
        
        invertible_constraint = tf.reduce_sum(tf.square(tf.matmul(transformation_A, transformation_A_bar) - tf.eye(REP_DIM))) + \
                                tf.reduce_sum(tf.square(tf.matmul(transformation_A_bar, transformation_A) - tf.eye(REP_DIM)))
                            
        weighted_src_loss = (1/num_pts) * tf.reduce_sum(tf.math.multiply(tf.math.multiply(prior_D_softmax, 1./prior_source), cw_src_loss))
        
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

selected_src_clss = []
for i in range(SRC_NUM_CLASSES):
    clss = np.argwhere(np.argmax(SRC_y_train_one_hot, 1) == i).flatten()
    selected_src_clss.append(clss)
    
selected_trg_clss = []
for i in range(TRG_NUM_CLASSES):
    clss = np.argwhere(np.argmax(TRG_y_train_one_hot, 1) == i).flatten()
    selected_trg_clss.append(clss)

cw_source_train_rep_loss = np.zeros(SRC_NUM_CLASSES)
cw_source_test_rep_loss = np.zeros(SRC_NUM_CLASSES)

for i in range(SRC_NUM_CLASSES):
    c_idx = np.argwhere(np.argmax(SRC_y_train_one_hot,1) == i).flatten()
    
    pred = SRC_net.classify(torch.tensor(SRC_train_rep_all[c_idx]).to(DEVICE)).cpu().detach().numpy()
    loss = np.sum(ce_loss_none(SRC_y_train_one_hot[c_idx], pred).numpy())
    cw_source_train_rep_loss[i] = loss

for i in range(SRC_NUM_CLASSES):
    c_idx = np.argwhere(np.argmax(SRC_y_test_one_hot,1) == i).flatten()
    
    pred = SRC_net.classify(torch.tensor(SRC_test_rep_all[c_idx]).to(DEVICE)).cpu().detach().numpy()
    loss = np.sum(ce_loss_none(SRC_y_test_one_hot[c_idx], pred).numpy())
    cw_source_test_rep_loss[i] = loss

for epoch in range(EPOCHS):
    
    b = transformation_b.numpy()
    A = transformation_A.numpy()
    A_bar = transformation_A_bar.numpy()
    B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
    D = tf.nn.softmax(prior_D).numpy()
    
    if epoch % 1 == 0:
        source_train_rep, source_y_train = get_set_based_on_prior_with_clss(SRC_train_rep_all, SRC_y_train_one_hot, BATCH_SIZE, D, SRC_NUM_CLASSES, selected_src_clss)
        modified_source_train_rep = (np.matmul(A, source_train_rep.T) + b).T
        modified_source_train_labels = tf.matmul(source_y_train, B.T).numpy()
    
    target_train_rep, target_y_train = get_set_based_on_prior_with_clss(TRG_train_rep_all, TRG_y_train_one_hot, BATCH_SIZE, prior_target, TRG_NUM_CLASSES, selected_trg_clss)
    
    modified_target_train_rep = target_train_rep
    
    full_feature_cost = torch.cdist(torch.tensor(modified_source_train_rep, device='cuda'), 
                                    torch.tensor(modified_target_train_rep, device='cuda')).cpu().numpy()
    
    full_label_cost = torch.cdist(torch.tensor(modified_source_train_labels, device='cuda'), 
                                    torch.tensor(target_y_train, device='cuda')).cpu().numpy()
    
    full_cost = full_feature_cost + LAMBDA * full_label_cost
    
    full_wasserstein_mapping = ot.emd(ot.unif(modified_source_train_rep.shape[0]), ot.unif(target_train_rep.shape[0]), full_cost)
    
    l, wd, ent, pm, ic, ws = optimize_WD(tf.convert_to_tensor(source_train_rep, tf.float32), 
                                         tf.convert_to_tensor(source_y_train, tf.float32), 
                                         tf.convert_to_tensor(target_train_rep, tf.float32), 
                                         tf.convert_to_tensor(target_y_train, tf.float32), 
                                         tf.convert_to_tensor(full_wasserstein_mapping, tf.float32), 
                                         tf.convert_to_tensor(cw_source_train_rep_loss, tf.float32),
                                         tf.convert_to_tensor(len(SRC_train_rep_all), tf.float32)
                                         )
    if epoch%100 == 0:
        
        b = transformation_b.numpy()
        A = transformation_A.numpy()
        A_bar = transformation_A_bar.numpy()
        B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
        D = tf.nn.softmax(prior_D).numpy()
        
        print(epoch, CHECKPOINT_PATH, "IMAGENET:", IMAGENET,
              "WD:", np.round(wd.numpy(),3),
              "CE:", np.round(ent.numpy(),3),
              "PM:", np.round(pm.numpy(),3),
              "IC:", np.round(ic.numpy(),3),
              "WS:", np.round(ws.numpy(),3),
              "ACC:", eval_accuracy_and_loss_cls_with_B_pytorch(np.matmul(A_bar, TRG_test_rep_all.T - b).T, TRG_y_test_one_hot, SRC_net, B, DEVICE, transpose_B = True),
              )
    
    if (epoch!=0 and epoch%1000 == 0) or (epoch == EPOCHS - 1):
        print("Plotting")
        b = transformation_b.numpy()
        A = transformation_A.numpy()
        A_bar = transformation_A_bar.numpy()
        B = tf.transpose(tf.nn.softmax(tf.transpose(B_matrix), axis=-1)).numpy()
        D = tf.nn.softmax(prior_D).numpy()
        
        CE = -tf.reduce_sum(D * tf.reduce_sum(B * tf.math.log(B), axis=0)).numpy()
        PM = tf.reduce_sum(tf.square(prior_target - np.matmul(B, D.reshape([-1,1])).reshape([-1]))).numpy()
        
        test_accs = eval_accuracy_and_loss_cls_with_B_pytorch(np.matmul(A_bar, TRG_test_rep_all.T - b).T, TRG_y_test_one_hot, SRC_net, B, DEVICE, transpose_B = True)
        
        modified_source_test_data_sample = (np.matmul(A, SRC_test_rep_all.T) + b).T
        modified_source_test_labels_sample = tf.matmul(SRC_y_test_one_hot, B.T).numpy()
        
        transformed_src_accs = eval_accuracy_and_loss_cls_with_B_pytorch(np.matmul(A_bar, modified_source_test_data_sample.T - b).T, modified_source_test_labels_sample, SRC_net, B, DEVICE, transpose_B = True)
       
        weigted_SRC_loss = (1/len(SRC_test_rep_all)) * tf.reduce_sum(tf.math.multiply(tf.math.multiply(D, 1./prior_source), cw_source_test_rep_loss)).numpy()
        
        WD_test = compute_WD_same_label_sets(modified_source_test_data_sample, 
                                             modified_source_test_labels_sample, 
                                             TRG_test_rep_all, 
                                             TRG_y_test_one_hot,
                                             1)
        
        
        modified_src_rep = (np.matmul(A, SRC_test_rep_all.T) + b).T
        modified_src_y = tf.matmul(SRC_y_test_one_hot, B.T).numpy()
        TAU = get_grad_norm_pytorch(np.matmul(A_bar, modified_src_rep.T - b).T, modified_src_y, 
                                    np.matmul(A_bar, TRG_test_rep_all.T - b).T, TRG_y_test_one_hot,
                                    SRC_net, B, DEVICE, TRG_NUM_CLASSES)
        
        print("Resnet18 SRC:", SRC, "TRG:", TRG, "Target test Accs: ", 
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
              np.round(TAU, 3),
              "\n")

print("################ END ################")
