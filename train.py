import time
import torch.nn.functional as F
from model_settings import args
from data_preprocessing import features, features_phon, labels, adj 
from data_preprocessing import idx_train, idx_val, idx_test 
from Model import GCN
import torch.optim as optim

# Create Model and Optimizer
model = GCN(nfeat=features.shape[1],   
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nhid3=args.hidden3,
            nclass=int(labels.max().item() + 1),
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

model_phon = GCN(nfeat=features_phon.shape[1],  
                  nhid1=args.hidden1,
                  nhid2=args.hidden2,
                  nhid3=args.hidden3,
                  nclass=int(labels.max().item() + 1),
                  dropout=args.dropout)
optimizer_phon = optim.Adam(model_phon.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

# Put them into cuda() if GPU available
if args.cuda:
    model.cuda()
    model_phon.cuda()
    features.cuda()
    features_phon.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    

def accuracy(output, labels):
    '''
    Get the accuracy of Model's prediction

    Parameters
    ----------
    output : TYPE float
        DESCRIPTION. accuracy of prediction [0, 1]
    labels : TYPE
        DESCRIPTION. labels of patients

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    preds = output.max(1)[1].type_as(labels)    # torch.max(1)[1]， 只返回每一行最大值的索引
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_model_Com(epoch):
    '''
    Train ComparE features Model

    Parameters
    ----------
    epoch : TYPE
        DESCRIPTION. Training Epochs

    Returns
    -------
    None.

    '''
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #loss_train = torch.nn.NLLLoss(output[idx_train], labels[idx_train].long())
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
    #loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train].long())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run. 在验证运行期间停用dropout
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val].long())
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
def test1():
    '''
    Test accuracy of Test dataset

    Returns
    -------
    None.

    '''
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test].long())
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    
def train_model_phon(epoch):
    '''
    Train Phonation_static features Model

    Parameters
    ----------
    epoch : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    t = time.time()
    model_phon.train()
    optimizer_phon.zero_grad()
    output = model_phon(features_phon, adj)
    #loss_train = torch.nn.NLLLoss(output[idx_train], labels[idx_train].long())
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].long())
    #loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train].long())
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer_phon.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run. 在验证运行期间停用dropout
        model_phon.eval()
        output = model_phon(features_phon, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val].long())
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test2():
    model_phon.eval()
    output = model_phon(features_phon, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test].long())
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    

    

