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
    def __init__(self,model,logger=None,taskcla=None, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)


        print('DIL CONTEXTUAL + KIM HAT')

        return


    def train(self,t,train,valid,num_train_steps,train_data,valid_data):

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


        best_loss=np.inf
        best_model=utils.get_model(self.model)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train
            clock0=time.time()
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step=self.train_epoch(t,train,iter_bar, optimizer,t_total,global_step)
            clock1=time.time()
            train_loss,train_acc,train_f1_macro=self.eval(t,train,trained_task=t)
            clock2=time.time()
            # print('time: ',float((clock1-clock0)*30*25))

            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                1000*self.args.train_batch_size*(clock1-clock0)/len(train),1000*self.args.train_batch_size*(clock2-clock1)/len(train),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc,valid_f1_macro=self.eval(t,valid,trained_task=t)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_model=utils.get_model(self.model)
                patience=self.lr_patience
                print(' *',end='')

            print()
            # break
        # Restore best
        utils.set_model_(self.model,best_model)

        # Activations mask
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        if t==0:
            self.mask_pre=mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals


        return

    def train_epoch(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        # Loop batches
        for step, batch in enumerate(iter_bar):
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, segment_ids, input_mask, targets,_= batch
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            # print('tokens_term_ids: ',tokens_term_ids)
            # Forward
            output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,s=s)
            masks=output_dict['masks']

            if 'dil' in self.args.scenario:
                output=output_dict['y']
            elif 'til' in self.args.scenario:
                outputs=output_dict['y']
                output = outputs[t]

            loss,_=self.criterion_hat(output,targets,masks)
            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den


            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            optimizer.step()

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            #print(masks[-1].data.view(1,-1))
            #if i>=5*self.sbatch: sys.exit()
            #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())
        #print(masks[-2].data.view(1,-1))

        return

    def eval(self,t,data,test=None,trained_task=None):
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

                if 'dil' in self.args.scenario:

                    if self.args.last_id: # fix 0
                        task=torch.LongTensor([trained_task]).cuda()
                        output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,s=self.smax)
                        output = output_dict['y']
                        masks = output_dict['masks']

                    #but don't know which one ========
                    # this is parameter-isolation method, kind of like multi-head
                    elif self.args.ent_id:
                        output_d= self.ent_id_detection(trained_task,input_ids, segment_ids, input_mask,t=t)
                        output = output_d['output']
                        masks = output_d['masks']

                elif 'til' in self.args.scenario:
                    task=torch.LongTensor([t]).cuda()
                    output_dict=self.model.forward(task,input_ids, segment_ids, input_mask,s=self.smax)
                    outputs = output_dict['y']
                    masks = output_dict['masks']
                    output = outputs[t]

                #but don't know which one ========


                loss,reg=self.criterion_hat(output,targets,masks)
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
