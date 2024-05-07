
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LogSoftmax, Sigmoid, Softmax
from torch.optim import Adam
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch import Tensor, argmax
import copy
import logging
import torch
from imbalanced_data.code.util import utils

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_name, loss_fct_name):
        super(MLP, self).__init__()

        if activation_name == "sigmoid":
            self.hidden_activation = F.sigmoid
        elif activation_name == "relu":
            self.hidden_activation = F.relu
        else:
            self.hidden_activation = F.relu

        self.loss_fct_name = loss_fct_name

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def __str__(self):
        return 'default MLP'

    def forward(self, data):
        # hidden layer ?? (input layer?)
        data = self.fc1(data)

        # hidden layer activation
        self.hidden_activation(data)

        # output layer
        target = self.fc2(data)

        # output activation
        # sklearn out activation for binary classification is LOGISTIC sigmoid
        # target = F.sigmoid(target)

        # target = softmax(target, dim=1)

        if self.loss_fct_name == "nlll":
            target = F.log_softmax(target, dim=1)
        if self.loss_fct_name == "BCE":
            target = F.sigmoid(target)

        return target

        # return sigmoid(self.fc2(self.hidden_activation(self.fc1(data))))

class MLPLoader:
    def __init__(self, config: dict,dataset: str, training_method: str, input_size: int, maj_miss_cost: float, min_miss_cost: float, imb_ratio: float):

        self.mlpConfig = config["training"]["MLP"] if training_method == "MLP" else config["training"]["MLP-Cost"]

        # Initializing Cuda
        self.device_name = self.mlpConfig["device_id"]
        self.device = utils.init_cuda(self.mlpConfig["device_id"],
                                      self.mlpConfig["torch_seed"])
        # logging.info('MLP on pytorch: Device_id = {}'.format(self.device))

        # model config
        self.input_size = input_size
        self.hidden_size = 100
        self.output_size = 1 if self.mlpConfig["loss_function"] == "BCE" or self.mlpConfig["loss_function"] == "BCEWithLogitsLoss" else 2

        # optimizer config
        self.learning_rate = 0.001
        self.betas = (0.9, 0.999)
        self.epsilon = 1e-08
        self.weight_decay_rate = 1e-4 # default = 0, l2 learning penalty -> linear lr decay

        # train config
        self.num_epoch = 200
        self.batch_size = 200
        self.tol = 1e-4#1e-4
        self.n_iter_no_change = 10

        self.model = MLP(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         output_size=self.output_size,
                         activation_name=self.mlpConfig["activation"],
                         loss_fct_name=self.mlpConfig["loss_function"])

        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(),
                              lr=self.learning_rate,
                              betas=self.betas,
                              eps=self.epsilon,
                              weight_decay=self.weight_decay_rate)

        # Lr Scheduler for learning rate decay at each step -> exponential lr decay

        if self.mlpConfig["activation"] == "sigmoid":
            logging.info("hidden activation: sigmoid")
        elif self.mlpConfig["activation"] == "relu":
            logging.info("hidden activation: relu")
        else:
            logging.info("hidden activation: relu")

# TODO test with other reduction method on loss or optimizer!!!!!!

        # loss function selection
        if training_method == "MLP":
            if self.mlpConfig["loss_function"] == "nlll":
                loss_fct = nn.NLLLoss()
                logging.info("loss function nlll")
            elif self.mlpConfig["loss_function"] == "cross":
                loss_fct = nn.CrossEntropyLoss()
                logging.info("loss function cross entropy")
            elif self.mlpConfig["loss_function"] == "BCE":
                loss_fct = nn.BCELoss()
                logging.info("loss function BCE")
            elif self.mlpConfig["loss_function"] == "BCEWithLogitsLoss":
                loss_fct = nn.BCEWithLogitsLoss()
                logging.info("loss function BCEWithLogitsLoss")

        if training_method == "MLP-Cost" or training_method == "MLP-Cost-IR":
            # false positive (truck being checked, but doesnt have falure in APS = 10 # negative klasse falsch klassifiziert
            # false negative (truck with failure was not checked) = 500 # positive klasse falsch klassifiziert
            # positive weight = 500/510
            # negative_weight = 10/510
            if self.mlpConfig["loss_function"] == "nlll":
                loss_fct = nn.NLLLoss(weight=Tensor([min_miss_cost, maj_miss_cost]).to(self.device))
                logging.info("loss function nlll with weights")
            elif self.mlpConfig["loss_function"] == "BCEWithLogitsLoss":
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=Tensor([imb_ratio]).to(self.device))
                logging.info("loss function BCE with logits and weights")



        self.loss = loss_fct

    def fit(self, x_npy, y_npy):

        inputs = Tensor(x_npy)
        targets = Tensor(y_npy)
        training_data = TensorDataset(inputs, targets)

        data_loader_params = { 'batch_size': self.batch_size,
                               'drop_last': False,
                               'shuffle': True,
                               'num_workers': 0}# maybe higher value

        data_loader = DataLoader(training_data, **data_loader_params)

        lowest_avg_loss = float("inf")
        loss_reset_counter = 0

        self.model.train()
        for epoch in range(self.num_epoch):
            cum_loss = 0
            count_elem = 0
            for batch_idx, (data, target) in enumerate(data_loader):

                tr_x = data.to(self.device).to(dtype=torch.float32)

                if self.mlpConfig["loss_function"] == "BCE" or self.mlpConfig["loss_function"] == "BCEWithLogitsLoss":
                    tr_y = target.to(self.device).to(dtype=torch.float32).reshape(-1, 1)
                elif self.mlpConfig["loss_function"] == "nlll" or self.mlpConfig["loss_function"] == "cross":
                    tr_y = target.to(self.device).to(dtype=torch.int64)

                pred = self.model(tr_x)

                loss = self.loss(pred, tr_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cum_loss += loss.item()
                count_elem += len(tr_y)

            avg_loss = cum_loss / count_elem
            if epoch % 10 == 0:
                print('Epoch [%d/%d], Loss: %.10f'
                      % (epoch + 1, self.num_epoch, avg_loss))

            #check if loss score does not get better (smaller) for 10 subsequent iterations
            if avg_loss > lowest_avg_loss - self.tol:
                loss_reset_counter += 1
            else:
                loss_reset_counter = 0
            if avg_loss < lowest_avg_loss:
                best_epoch = epoch
                lowest_avg_loss = avg_loss

            best_model_state = copy.deepcopy(self.model.state_dict())

            if loss_reset_counter >= 10:
                break


        logging.info('Most successful epoch = {:2d} | lowest loss : {}'.format(best_epoch, lowest_avg_loss))
        self.model.load_state_dict(best_model_state)
        logging.info('Training done!')

    # sklearn default treshhold for binary classification is 0.5
    # https://stackoverflow.com/questions/19984957/scikit-learn-predict-default-threshold
    def predict(self, x):
        self.model.eval()
        y_pred = self.model(Tensor(x).to(self.device).type(torch.float32))

        if self.mlpConfig["loss_function"] == "BCE":
            # (output 1)
            y_pred = y_pred.round().cpu().detach().numpy()
        elif self.mlpConfig["loss_function"] == "BCEWithLogitsLoss":
            # (output 1)
            y_pred = nn.Sigmoid()(y_pred).round().cpu().detach().numpy()
        else:
            # choose class with highest probability for prediciton (output 2)
            y_pred = argmax(y_pred, dim=1).cpu().detach().numpy().astype(int)
        return y_pred

    def predict_proba(self, x):
        self.model.eval()
        y_pred = self.model(Tensor(x).to(self.device).type(torch.float32))
        y_pred = Softmax(dim=1)(y_pred)
        return y_pred.cpu().detach().squeeze().numpy()

    def get_model(self):
        return self.model

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def parameters(self):
        return self.model.parameters()
