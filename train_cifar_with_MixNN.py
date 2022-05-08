import torch
import random
from MixNN_Final_Source_Code import resnet
from sklearn.mixture import GaussianMixture
import torch.optim as optim
from torchvision import datasets
import os
import sys
sys.path.append('..')
import numpy as np
from MixNN_Final_Source_Code.dataloader_cifar import cifar_dataloader
import argparse
import torch.nn.functional as F
from MixNN_Final_Source_Code.loss import ANNLoss_K_2, SCELoss
import nmslib

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym',
                    help='sym -> symmetric label noise and asym -> asymmetric label noise')
parser.add_argument('--r', default=0.8, type=float, help='noise ratio r in (0,1)')
parser.add_argument('--model', default='resnet34', type=str)
parser.add_argument('--op', default='SGD', type=str, help='optimizer')
parser.add_argument('--es', default=60, help='the epoch starts update target')
parser.add_argument('--warmup', default=10, help='warm up epochs, 10 for cifar10, 30 for cifar100')
parser.add_argument('--m', default=0.9, type=float, help='the momentum parameter in target update.')
parser.add_argument('--lr_s', default='CosineAnnealingWarmRestarts', type=str, help='learning rate scheduler')
parser.add_argument('--loss', default='ANNLoss', type=str, help='loss function')
parser.add_argument('--K', default=2, help='K ANN')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--log_interval', default=100, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='..', type=str,
                    help='path to save dataset and results')
parser.add_argument('--dataset', default='cifar10', type=str)


args = parser.parse_args()

assert args.num_epochs > args.es
if args.dataset == 'cifar10':
    print('############## Dataset CIFAR-10 ######################')
    num_class = 10
    _ = datasets.CIFAR10(root=args.data_path, train=True, download=True)
    args.data_path = os.path.join(args.data_path, 'cifar-10-batches-py')
elif args.dataset == 'cifar100':
    num_class = 100
    print('############## Dataset CIFAR-100 ######################')
    _ = datasets.CIFAR100(root=args.data_path, train=True, download=True)
    args.data_path = os.path.join(args.data_path, 'cifar-100-python')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.seed:
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation


def mixup_data_ann_dynamic_K_2(x, y, x_feature, clean_prob, args, epoch):
    batch_size = x.size()[0]
    if epoch > args.es:
        # todo: index should be the original data's nearest neighbor
        nn_index = nmslib.init(method='hnsw', space='l2')
        feature_np = x_feature.cpu().detach().numpy()
        nn_index.addDataPointBatch(feature_np)
        nn_index.createIndex({'post': 2}, print_progress=False)
        re = nn_index.knnQueryBatch(feature_np, k=2, num_threads=4)
        index = [item[0][1] for item in re]  # nn index in each batch
        index = torch.from_numpy(np.array(index)).cuda()
        index = index.type(torch.int64)  # only int64 can be used as indices
        lam_ori = clean_prob
        lam_aug = clean_prob[index.cpu().numpy()]
        lam_ori = torch.from_numpy(lam_ori)  # original object clean probability
        lam_aug = torch.from_numpy(lam_aug)  # augment object clean probability
        lam = (lam_ori + 1e-10) / (lam_ori + lam_aug + 1e-10)
        x_nn = x[index, :]
        lam = lam.float().cuda()
        for i in range(len(x)):
            x[i] = lam[i] * x[i] + (1. - lam[i]) * x_nn[i]
        y_a, y_b = y, y[index]  # y_a is the original label, y_b is the label for mixed data
        return x, y_a, y_b, lam, index

    else:
        # index is random
        lam = np.random.beta(1, 1)
        if args.gpuid != -1:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]  # y_a is the original label, y_b is the label for mixed data
        return mixed_x, y_a, y_b, lam, index

def train(args, model, train_loader, noisy_index, probs, optimizer, epoch):
    model.train()
    loss_per_batch = []
    correct = 0
    n_correct = 0
    c_correct = 0
    n_num = 0
    c_num = 0
    acc_train_per_batch = []
    acc_train_noisy_per_batch = []
    acc_train_clean_per_batch = []
    for batch_idx, (data, target, index) in enumerate(train_loader):
        data, target = data.to(args.gpuid), target.to(args.gpuid)
        optimizer.zero_grad()
        output_ori, features, _ = model(data)

        if args.K == 2:
            inputs, targets_a, targets_b, lam, nn_index_in_batch = mixup_data_ann_dynamic_K_2(data, target, features,
                                                                                              probs[index],
                                                                                              args, epoch)
            inputs, targets_a, targets_b = inputs.to(args.gpuid), targets_a.to(args.gpuid), targets_b.to(
                args.gpuid)

            output, _, _ = model(inputs)

            loss = criterion(output, output_ori, targets_a, targets_b, index, nn_index_in_batch, lam, epoch)

        loss.backward(retain_graph=True)
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        n_index = noisy_index[index]
        n_num += np.sum(n_index)
        n_correct += pred[n_index].eq(target[n_index].view_as(pred[n_index])).sum().item()
        c_index = np.logical_not(noisy_index[index])
        c_num += np.sum(c_index)
        c_correct += pred[c_index].eq(target[c_index].view_as(pred[c_index])).sum().item()

        acc_train_noisy_per_batch.append(100. * n_correct / n_num)
        acc_train_clean_per_batch.append(100. * c_correct / c_num)
        if batch_idx % args.log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Accuracy: {:.0f}%, noisy Accuracy: {} / {} = {:.0f}%, clean Accuracy: {} / {} = {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(),
                           100. * correct / ((batch_idx + 1) * args.batch_size),
                    n_correct, n_num,
                           100. * n_correct / n_num,
                    c_correct, c_num,
                           100. * c_correct / c_num,
                    optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    print(
        'Train Epoch: {}, fit noisy / total noisy = {} / {} = {}, fit clean / total clean = {} /{} ={}'.format(epoch,
                                                                                                               n_correct,
                                                                                                               n_num,
                                                                                                               n_correct / n_num,
                                                                                                               c_correct,
                                                                                                               c_num,
                                                                                                               c_correct / c_num))
    # acc_train_noisy_per_batch = [np.average(acc_train_noisy_per_batch)]
    # acc_train_clean_per_batch = [np.average(acc_train_clean_per_batch)]
    return loss_per_epoch, acc_train_per_epoch, n_correct / n_num, c_correct / c_num


def test_cleaning(test_batch_size, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return loss_per_epoch, acc_val_per_epoch

def eval_train_gmm(model, all_loss, corrected_targets):
    model.eval()
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _, _ = model(inputs)
            # optional, use the original target or corrected target
            # loss = CE(outputs, targets)
            loss = CE(outputs, corrected_targets[index])
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())  # normalize the loss
    all_loss.append(losses)

    if args.r == 0.8:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)
        # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss

class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def warmup(epoch, model, optimizer, dataloader):
    model.train()
    for batch_idx, (data, target, index) in enumerate(dataloader):
        data, target = data.to(args.gpuid), target.to(args.gpuid)
        optimizer.zero_grad()
        output, _, _ = model(data)
        loss = CEloss(output, target)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, batch_idx,
                            loss.item()))
        sys.stdout.flush()

exp_path = os.path.join('./',
                        '{0}_noise_models_{1}_{2}_K={3}_{4}_{5}_{6}_bs={7}'.format(
                            args.dataset,
                            args.model,
                            args.loss,
                            args.K,
                            args.op,
                            args.lr_s, args.num_epochs,
                            args.batch_size),
                        args.noise_mode + str(args.r) + '_seed=' + str(args.seed))
if not os.path.isdir(exp_path):
    os.makedirs(exp_path)

loader = cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                          num_workers=5,
                          root_dir=args.data_path,
                          noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

all_trainloader, noisy_labels, clean_labels = loader.run('train')
test_loader = loader.run('test')
eval_train_loader, _, _ = loader.run('eval_train')
noisy_index = (noisy_labels != clean_labels)

if args.model == 'resnet34':
    model = resnet.ResNet34(num_classes=args.num_class).to(args.gpuid)

if args.op == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

if args.lr_s == 'CosineAnnealingWarmRestarts':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.001)

CE = torch.nn.CrossEntropyLoss(reduction='none')
CEloss = torch.nn.CrossEntropyLoss()
if args.K == 2:
    criterion = ANNLoss_K_2(noisy_labels, args.num_class, args.es, args.m)

all_loss = [[]]
cont = 0
acc_train_per_epoch_model = np.array([])
acc_train_noisy_per_epoch_model = np.array([])
acc_train_clean_per_epoch_model = np.array([])
loss_train_per_epoch_model = np.array([])
acc_val_per_epoch_model = np.array([])
loss_val_per_epoch_model = np.array([])
for epoch in range(1, args.num_epochs + 1):

    if epoch < args.warmup:
        print('\t##### Warmup #####')
        warmup(epoch, model, optimizer, eval_train_loader)
    else:
        print('\t##### Training #####')
        _, corrected_labels = torch.max(criterion.soft_labels, dim=1)
        prob, all_loss[0] = eval_train_gmm(model, all_loss[0], corrected_labels)

        loss_train_per_epoch, acc_train_per_epoch, acc_train_noisy_per_epoch, acc_train_clean_per_epoch = train(args,
                                                                                                                model,
                                                                                                                all_trainloader,
                                                                                                                noisy_index,
                                                                                                                prob,
                                                                                                                optimizer,
                                                                                                                epoch)
        loss_per_epoch, acc_val_per_epoch_i = test_cleaning(args.batch_size, model, args.gpuid, test_loader)
        acc_train_per_epoch_model = np.append(acc_train_per_epoch_model, acc_train_per_epoch)
        acc_train_noisy_per_epoch_model = np.append(acc_train_noisy_per_epoch_model, acc_train_noisy_per_epoch)
        acc_train_clean_per_epoch_model = np.append(acc_train_clean_per_epoch_model, acc_train_clean_per_epoch)
        loss_train_per_epoch_model = np.append(loss_train_per_epoch_model, loss_train_per_epoch)
        acc_val_per_epoch_model = np.append(acc_val_per_epoch_model, acc_val_per_epoch_i)
        loss_val_per_epoch_model = np.append(loss_val_per_epoch_model, loss_per_epoch)
        if epoch == 10:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        elif epoch > 10:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]
                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                        # os.remove(os.path.join(exp_path, lossBest))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                    epoch, loss_per_epoch[-1], acc_val_per_epoch_i[-1], args.r, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1
    scheduler.step()

np.save(os.path.join(exp_path, 'acc_train_per_epoch_model.npy'), acc_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_train_noisy_per_epoch_model.npy'), acc_train_noisy_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_train_clean_per_epoch_model.npy'), acc_train_clean_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_train_per_epoch_model.npy'), loss_train_per_epoch_model)
np.save(os.path.join(exp_path, 'acc_val_per_epoch_model.npy'), acc_val_per_epoch_model)
np.save(os.path.join(exp_path, 'loss_val_per_epoch_model.npy'), loss_val_per_epoch_model)

