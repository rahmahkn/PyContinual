import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
from buffer import Buffer as Buffer
# from apex import amp

import torch.autograd as autograd
sys.path.append("./approaches/")
from contrastive_loss import SupConLoss, CRDLoss
from copy import deepcopy

class Appr(object):

    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x/warmup
        return 1.0 - x


    def __init__(self,model,logger, taskcla,args=None):

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.model=model
        self.nepochs=args.nepochs
        self.lr=args.lr
        self.lr_min=args.lr_min
        self.lr_factor=args.lr_factor
        self.lr_patience=args.lr_patience
        self.clipgrad=args.clipgrad
        self.args = args

        self.train_batch_size=args.train_batch_size
        self.eval_batch_size=args.eval_batch_size
        self.args=args
        self.ce=torch.nn.CrossEntropyLoss()
        self.sup_con = SupConLoss(temperature=args.temp,base_temperature=args.base_temp)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        
        logger.info("device: {} n_gpu: {}".format(
            self.device, self.n_gpu))

        if 'one' in args.baseline:
            self.initial_model=deepcopy(model)

        if  args.baseline=='a-gem':
            self.buffer = Buffer(self.args.buffer_size, self.device)
            self.grad_dims = []
            for param in self.model.parameters():
                self.grad_dims.append(param.data.numel())
            self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.device)
            self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.device)

            self.grads_cs = []
            
        if args.baseline=='ewc':
            self.lamb=args.lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000
            self.fisher=None
            
        if  args.baseline=='hat':
            self.smax = 400  # Grid search = [140,200,300,400]; best was 400
            self.thres_cosh=50
            self.thres_emb=6
            self.lamb=0.75
            self.mask_pre=None
            self.mask_back=None

        print('BERT NCL')

        return

    def sup_loss(self,output,pooled_rep,input_ids, segment_ids, input_mask,targets,t):
        if self.args.sup_head:
            outputs = torch.cat([output.clone().unsqueeze(1), output.clone().unsqueeze(1)], dim=1)
        else:
            outputs = torch.cat([pooled_rep.clone().unsqueeze(1), pooled_rep.clone().unsqueeze(1)], dim=1)

        loss = self.sup_con(outputs, targets,args=self.args)
        return loss


    def f1_compute_fn(self,y_true, y_pred,average):
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise RuntimeError("This contrib module requires sklearn to be installed.")

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred,average=average)

    def project(self,gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
        corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
        return gxy - corr * ger


    def store_grad(self,params, grads, grad_dims):
        """
            This stores parameter gradients of past tasks.
            pp: parameters
            grads: gradients
            grad_dims: list with number of parameters per layers
        """
        # store the gradients
        grads.fill_(0.0)
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = np.sum(grad_dims[:count + 1])
                grads[begin: end].copy_(param.grad.data.view(-1))
            count += 1


    def overwrite_grad(self,params, newgrad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        count = 0
        for param in params():
            if param.grad is not None:
                begin = 0 if count == 0 else sum(grad_dims[:count])
                end = sum(grad_dims[:count + 1])
                this_grad = newgrad[begin: end].contiguous().view(
                    param.grad.data.size())
                param.grad.data.copy_(this_grad)
            count += 1


    def _get_optimizer_owm(self, lr=None):
        # if lr is None:
        #     lr = self.lr
        lr = self.lr
        lr_owm = self.lr
        fc1_params = list(map(id, self.model.fc1.parameters()))
        fc2_params = list(map(id, self.model.fc2.parameters()))
        base_params = filter(lambda p: id(p) not in fc1_params + fc2_params,
                             self.model.parameters())
        optimizer = torch.optim.SGD([{'params': base_params},
                                     {'params': self.model.fc1.parameters(), 'lr': lr_owm},
                                     {'params': self.model.fc2.parameters(), 'lr': lr_owm},
                                     ], lr=lr, momentum=0.9)

        return optimizer

    def _get_optimizer_ucl(self, lr=None, lr_rho = None):
        if lr is None: lr = self.lr
        if lr_rho is None: lr_rho = self.lr_rho
        if self.args.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, lr_rho=lr_rho, param_name = self.param_name)
        if self.args.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),lr=lr)


    def _get_optimizer_kan(self,lr=None,which_type=None):

        if which_type=='mcl':
            if lr is None: lr=self.lr
            if self.args.optimizer == 'sgd':
                return torch.optim.SGD(
                    [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
            elif self.args.optimizer == 'adam':
                return torch.optim.Adam(
                    [p for p in self.model.mcl.parameters()]+[p for p in self.model.last.parameters()],lr=lr)

        elif which_type=='ac':
            if lr is None: lr=self.lr
            if self.args.optimizer == 'sgd':
                return torch.optim.SGD(
                    [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)
            elif self.args.optimizer == 'adam':
                    return torch.optim.Adam(
                        [p for p in self.model.ac.parameters()]+[p for p in self.model.last.parameters()],lr=lr)



    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.args.optimizer == 'sgd' and self.args.momentum:
            print('sgd+momentum')
            return torch.optim.SGD(self.model.parameters(),lr=lr, momentum=0.9,nesterov=True)
        elif self.args.optimizer == 'sgd':
            print('sgd')
            return torch.optim.SGD(self.model.parameters(),lr=lr)
        elif self.args.optimizer == 'adam':
            print('adam')
            return torch.optim.Adam(self.model.parameters(),lr=lr)


    def ent_id_detection(self,trained_task,input_ids, segment_ids, input_mask,t,which_type=None):

        output_d = {}

        outputs = []
        entropies = []

        if trained_task is None: #training
            entrop_to_test = range(0, t + 1)
        else: #testing
            entrop_to_test = range(0, trained_task + 1)

        for e in entrop_to_test:
            e_task=torch.LongTensor([e]).cuda()
            if 'hat' in self.args.baseline:
                output_dict = self.model.forward(e_task,input_ids, segment_ids, input_mask,s=self.smax)
                masks = output_dict['masks']
                output_d['masks']= masks

            elif 'kan' in self.args.baseline:
                output_dict = self.model.forward(e_task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
            output = output_dict['y']
            outputs.append(output) #shared head

            Y_hat = F.softmax(output, -1)
            entropy = -1*torch.sum(Y_hat * torch.log(Y_hat))
            entropies.append(entropy)
        inf_task_id = torch.argmin(torch.stack(entropies))
        output=outputs[inf_task_id]

        output_d['output']= output

        return output_d




    def criterion_hat(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()

        reg/=count
        return self.ce(outputs,targets)+self.lamb*reg,reg

    def criterion_ewc(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.ce(output,targets)+self.lamb*loss_reg

########################################################################################################################
class CheckFederated():
    def __init__(self):
        pass
    def set_similarities(self,similarities):
        self.similarities = similarities

    def fix_length(self):
        return len(self.similarities)

    def get_similarities(self):
        return self.similarities


    def check_t(self,t):
        if t < len([sum(x) for x in zip_longest(*self.similarities, fillvalue=0)]) and [sum(x) for x in zip_longest(*self.similarities, fillvalue=0)][t] > 0:
            return True

        elif np.count_nonzero(self.similarities[t]) > 0:
            return True

        elif t < len(self.similarities[-1]) and self.similarities[-1][t] == 1:
            return True

        return False