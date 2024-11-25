import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from utils.logger import CsvWriter
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb


def global_evaluate(model: FederatedModel, test_dl: DataLoader,iter, setting: str, name: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        if 'X' in locals():
            del X
            del Y
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
                if batch_idx == 0:
                    X = outputs
                    Y= labels
                else:
                    X = torch.cat((X, outputs),dim=0)
                    Y =  torch.cat((Y,labels),dim=0)
        if iter+1 ==5 or iter+1==15  or iter+1==50 or iter+1==75:
            model1 = TSNE(n_components=2, random_state=0)
            testx=X.cpu()
            testy=Y.cpu()
            if len(testy)<=30:
                testx=torch.cat((testx,testx),dim=0)
                testy=torch.cat((testy,testy),dim=0)
            transformed = model1.fit_transform(testx)
            testY=torch.rand(len(testy))
            for a in range(len(testy)):
                testY[a]=testy[a].item()
                # Plotting
            plt.figure(figsize=(8, 6))
            for class_value in range(65):
                    # Select points that belong to the current class
                ii = testY == class_value
                plt.scatter(transformed[ii, 0], transformed[ii, 1])

            plt.legend()
            plt.title(f't-SNE visualization of {j} dataset - Iteration {iter+1}')

            filename = f't_sne_{j}_iteration_{iter + 1}_{j}.png'
            plt.savefig(filename)
            plt.close()
        top1acc = round(100 * top1 / total, 2)
        top5acc = round(100 * top5 / total, 2)

    net.train(status)
    return accs

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    model.eval()
    with torch.no_grad():
        preds = model(grid_tensor)
        Z = preds.max(1)[1].reshape(xx.shape)
    model.train()
    plt.contourf(xx, yy, Z, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("Neural Network Decision Boundary")
    plt.show()

def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_digits':
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True,
                                                        p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_officehome':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            elif model.args.dataset == 'fl_domain_net':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_PACS':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True, p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:

        selected_domain_dict = {'caltech': 1, 'amazon': 1, 'webcam': 9, 'dslr': 9}  # 20

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []

    alpha = 1/args.parti_num * np.ones(args.parti_num).T
    Epoch = args.communication_epoch
    wandb.init(project="FPL",config={
    "learning_rate": 0.01,
    "architecture": "CNN",
    "dataset": "args.dataset",
    "epochs": "args.communication_epoch",
    })
    best_acc=0
    best_epoch=0
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index
        if hasattr(model, 'loc_update'):
            loss,theta,alpha = model.loc_update(pri_train_loaders)
        accs = global_evaluate(model, test_loaders,epoch_index, private_dataset.SETTING, private_dataset.NAME)
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        if epoch_index==0:
            consit_term = 10
            fl_loss = np.dot(alpha, loss)+consit_term
        else:
            consit_term = 0.5 * np.dot(alpha - theta[0], alpha - theta[0])
            fl_loss = np.dot(alpha, loss)
        global_loss=np.dot(alpha,loss)+consit_term
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]
        if mean_acc>best_acc:
            best_acc=mean_acc
            best_epoch=epoch_index
            wandb.save("model.pth")

        print('The ' + str(epoch_index) + ' Communcation Accuracy:', str(mean_acc), 'Method:', model.args.model, 'Global_Loss:', str(global_loss),'Best_epoch:', str(best_epoch), 'Best_acc:', str(best_acc))
        wandb.log({"epoch": epoch_index, "loss": global_loss, "accuracy": mean_acc, "fl_loss": fl_loss})
        print(accs)
        print(mean_acc)
        print(global_loss)
    wandb.finish()
    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)
