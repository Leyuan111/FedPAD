import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class fedpad:
    NAME = 'fedpad'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform, seed=0):
        self.nets_list = nets_list
        self.args = args
        self.transform = transform
        self.global_protos = {}
        self.local_protos = {}
        self.par_num = args.parti_num  # Number of clients
        self.N_CLASS = args.num_classes  # Number of classes
        self.local_epoch = args.local_epoch
        self.local_lr = args.local_lr
        self.random_state = np.random.RandomState(seed)
        self.online_clients = []
        self.global_net = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_to_device()
        self.ini()

        # Initialize theta as a trainable parameter
        self.theta = torch.nn.Parameter(torch.ones(self.par_num, device=self.device) / self.par_num)
        # Project theta onto the simplex
        self.theta.data = self.project_simplex(self.theta.data)

        # Define the optimizer to update theta
        self.theta_optimizer = optim.SGD([self.theta], lr=0.001)

        # Initialize local learning rates for each client
        self.local_lrs = [self.local_lr for _ in range(self.par_num)]
        # Set learning rate bounds
        self.min_lr = self.args.local_lr / 1000  # Adjust as needed
        self.max_lr = self.args.local_lr * 10    # Adjust as needed

        # Add properties to monitor theta changes
        self.theta_change_threshold = 1e-4  # Adjust threshold as needed
        self.stop_training = False  # Flag to indicate whether to stop training

    def net_to_device(self):
        """
        Move all network models to the specified device.
        """
        for net in self.nets_list:
            net.to(self.device)

    def ini(self):
        """
        Initialize the global model and synchronize local model parameters with the global model.
        """
        self.global_net = copy.deepcopy(self.nets_list[0])
        self.global_net.to(self.device)
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)
            net.to(self.device)

    def compute_alpha(self):
        """
        Compute alpha from theta: alpha_i = softmax(-theta_i / 0.1).
        """
        alpha = torch.softmax(-self.theta / 0.1 , dim=0)
        return alpha

    def loc_update(self, priloader_list):
        total_clients = list(range(self.par_num))
        self.online_clients = total_clients
        losses = np.zeros(self.par_num)
        losses1 = np.zeros(self.par_num)
        local_protos_list = [{} for _ in range(self.par_num)]

        # 1. Compute alpha from the current theta
        alpha = self.compute_alpha()

        # 2. Adjust clients' learning rates based on the new alpha before local training
        local_lrs_tensor = torch.tensor(self.local_lrs, device=self.device)
        global_lr = torch.sum(alpha * local_lrs_tensor).item()

        # Adjust clients' learning rates for the next round based on the new global_lr and clients' alpha
        new_local_lrs = [0] * self.par_num
        for i in self.online_clients:
            alpha_i = alpha[i].item()
            # Compute new learning rate for client i
            adaptive_lr = max(global_lr / 1000, global_lr * (1 + (1 / self.par_num - alpha_i) * self.par_num))
            # Apply learning rate bounds
            adaptive_lr = max(self.min_lr, min(self.max_lr, adaptive_lr))
            new_local_lrs[i] = adaptive_lr

        # Update clients' learning rates to be used in this round of local training
        self.local_lrs = new_local_lrs

        # 3. Clients perform local training using the updated learning rates
        for i in self.online_clients:
            # Get current learning rate for client (updated)
            current_lr = self.local_lrs[i]

            # Client training, returns updated learning rate
            loss, local_protos, loss1, updated_lr = self._train_net(
                i, self.nets_list[i], priloader_list[i], current_lr
            )
            losses1[i] = loss1
            losses[i] = loss
            local_protos_list[i] = local_protos

            # Since we have adjusted learning rates before training, we can keep the updated_lr
            # Or you can reassign it if your _train_net method adjusts learning rate further
            self.local_lrs[i] = updated_lr

        # 4. Compute the losses and update theta based on the clients' training results
        # Convert losses1 to a tensor without requires_grad
        losses_tensor1 = torch.tensor(losses1, device=self.device, requires_grad=False)

        # Compute total loss
        total_loss = torch.sum(losses_tensor1 * alpha)

        # Backpropagate and update theta
        self.theta_optimizer.zero_grad()
        total_loss.backward()
        self.theta_optimizer.step()

        # Project theta onto the simplex
        with torch.no_grad():
            self.theta.copy_(self.project_simplex(self.theta))

        # Compute theta change
        theta_change = torch.norm(self.theta - self.theta.detach(), p=2).item()
        print(f"Theta change: {theta_change}")

        freq = alpha.detach().cpu().numpy()
        print("Alpha:", alpha)
        print("Theta:", self.theta)
        self.aggregate_nets(freq)
        self.global_protos = self.new_proto_aggregation(local_protos_list, alpha.detach())

        return losses, self.theta.detach().cpu().numpy(), alpha.detach().cpu().numpy()

    def project_simplex(self, v):
        """
        Project vector v onto the simplex D = { x | x >= 0, sum x_i = 1 }.
        """
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - 1
        ind = torch.arange(len(v), device=self.device) + 1
        cond = v_sorted - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / rho
        w = torch.clamp(v - theta, min=0)
        return w

    def aggregate_nets(self, freq):
        state_dict_list = [net.state_dict() for net in self.nets_list]
        global_state_dict = {}

        for key in state_dict_list[0].keys():
            param = state_dict_list[0][key]
            if 'running_mean' in key or 'running_var' in key:
                # For BatchNorm stats, perform unweighted average
                global_state_dict[key] = torch.stack(
                    [state_dict[key].to(self.device) for state_dict in state_dict_list]
                ).mean(dim=0)
            else:
                # For other parameters, use freq for weighted average
                global_param = torch.zeros_like(param, dtype=param.dtype)
                for i in range(len(self.nets_list)):
                    freq_i = torch.tensor(freq[i], dtype=param.dtype, device=self.device)
                    global_param += freq_i * state_dict_list[i][key].to(self.device)
                global_state_dict[key] = global_param

        self.global_net.load_state_dict(global_state_dict)
        for net in self.nets_list:
            net.load_state_dict(self.global_net.state_dict())
            net.to(self.device)

    def new_proto_aggregation(self, local_protos_list, alpha):
        """
        Aggregate local prototypes to update global prototypes.
        """
        proto_sums = {}
        weight_sums = {}

        for idx in self.online_clients:
            local_protos = local_protos_list[idx]
            for label, proto in local_protos.items():
                weight = alpha[idx].item()
                if label in proto_sums:
                    proto_sums[label] += weight * proto
                    weight_sums[label] += weight
                else:
                    proto_sums[label] = weight * proto
                    weight_sums[label] = weight

        # Compute weighted average prototype for each label
        updated_global_protos = {
            label: (proto_sums[label] / weight_sums[label]).detach()
            for label in proto_sums
        }

        return updated_global_protos

    def _train_net(self, index, net, train_loader, current_lr):
        """
        Local training process, possibly adjusting the learning rate during training.
        """
        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(self.device)
        mse_loss_fn = nn.MSELoss().to(self.device)

        running_loss = 0.0
        local_protos = {}

        for epoch in range(self.local_epoch):
            usingloss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                images = images.to(self.device)
                labels = labels.to(self.device)
                features=net(images)
                outputs=features

                loss_CE = criterion(outputs, labels)
                loss = loss_CE

                # If global prototypes exist, compute MSE loss
                if self.global_protos:
                    mse_loss = 0.0
                    for label in labels.unique():
                        mask = labels == label
                        if label.item() in self.global_protos:
                            global_proto = self.global_protos[label.item()].to(self.device)
                            mse_loss += mse_loss_fn(features[mask], global_proto)
                    mse_loss = mse_loss / len(labels.unique())
                    loss +=1*mse_loss  # self.mu defaults to 1, adjust if needed

                # Collect local prototypes
                for label in labels.unique():
                    mask = labels == label
                    if label.item() not in local_protos:
                        local_protos[label.item()] = []
                    proto = features[mask].mean(dim=0).detach()
                    local_protos[label.item()].append(proto)

                # Backpropagate and update local model parameters
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
                optimizer.step()

                running_loss += loss.item()
                usingloss += loss.item()

            avg_usingloss = usingloss / len(train_loader)

        # After training, get the updated learning rate from the optimizer
        updated_lr = optimizer.param_groups[0]['lr']

        # Compute the average prototype for each class
        for label in local_protos:
            protos = torch.stack(local_protos[label])
            local_protos[label] = protos.mean(dim=0)

        avg_loss = running_loss / (self.local_epoch * len(train_loader))
        return avg_loss, local_protos, avg_usingloss, updated_lr
