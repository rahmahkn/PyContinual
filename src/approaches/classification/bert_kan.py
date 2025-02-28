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
# from apex import amp
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
import torch.autograd as autograd
sys.path.append("./approaches/base/")
from bert_base import Appr as ApprBase

class Appr(ApprBase):
    def __init__(self,model,args=None,logger=None,taskcla=None):
    # def __init__(self,model,nepochs=100,sbatch=64,lr=0.001,lr_min=1e-5,lr_factor=2,lr_patience=3,clipgrad=10000,args=None,logger=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        print('CONTEXTUAL + RNN NCL')

        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data):
        # self.model=deepcopy(self.initial_model) # Restart model: isolate


        global_step = 0
        self.model.to(self.device)

        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)

        if t == 0: which_types = ['mcl']
        else: which_types = ['ac','mcl']

        for which_type in which_types:

            print('Training Type: ',which_type)

            best_loss=np.inf
            best_model=utils.get_model(self.model)
            lr=self.lr
            patience=self.lr_patience
            self.optimizer=self._get_optimizer_kan(lr,which_type)

            # Loop epochs
            for e in range(self.args.num_train_epochs):
                # Train
                clock0=time.time()
                iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
                global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
                clock1=time.time()
                
                train_loss,train_acc,train_f1_macro=self.eval(t,train,train_data,which_type,trained_task=t)
                clock2=time.time()
                print('time: ',float((clock1-clock0)*30*25))

                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.args.train_batch_size*(clock1-clock0)/len(train),1000*self.args.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,valid_data,which_type,trained_task=t)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')

                print()

            # Restore best
            utils.set_model_(self.model,best_model)

        return



    def train_epoch(self,t,data,iter_bar,which_type):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=True)

            # Forward
            output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s)
            if 'dil' in self.args.scenario:
                output = output_dict['y']
            elif 'til' in self.args.scenario:
                outputs = output_dict['y']
                output = outputs[t]

            loss=self.ce(output,targets)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if t>0 and which_type=='mcl':
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
                mask=self.model.ac.mask(task,s=self.smax)
                mask = torch.autograd.Variable(mask.data.clone(),requires_grad=False)
                for n,p in self.model.named_parameters():
                    if n in rnn_weights:
                        # print('n: ',n)
                        # print('p: ',p.grad.size())
                        p.grad.data*=self.model.get_view_for(n,mask)

            # Compensate embedding gradients
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            optimizer.step()

            # Constrain embeddings
            for n,p in self.model.ac.named_parameters():
                if 'ac.e' in n:
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        return global_step

    def eval(self,t,data,test,which_type,trained_task):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()
        target_list = []
        pred_list = []
        with torch.no_grad():

            for step, batch in enumerate(data):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, segment_ids, input_mask, targets,_= batch
                real_b=input_ids.size(0)
                task=torch.LongTensor([trained_task]).cuda()

                if 'dil' in self.args.scenario:
                    if self.args.last_id: # fix 0
                        output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
                        output = output_dict['y']

                    elif self.args.ent_id:
                        output_d= self.ent_id_detection(trained_task,input_ids, segment_ids, input_mask,t,which_type)
                        output = output_d['output']

                elif 'til' in self.args.scenario:
                    task=torch.LongTensor([t]).cuda()
                    output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,which_type,s=self.smax)
                    outputs = output_dict['y']
                    output = outputs[t]


                loss=self.ce(output,targets)

                _,pred=output.max(1)
                hits=(pred==targets).float()
                target_list.append(targets)
                pred_list.append(pred)
                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_acc+=hits.sum().data.cpu().numpy().item()
                total_num+=real_b
            f1=self.f1_compute_fn(y_pred=torch.cat(pred_list,0),y_true=torch.cat(target_list,0),average='macro')

        return total_loss/total_num,total_acc/total_num,f1
