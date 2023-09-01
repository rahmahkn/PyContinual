import sys
import torch
from transformers import BertModel, BertConfig
import utils
from torch import nn
import torch.nn.functional as F

class Net(torch.nn.Module):

    def __init__(self,taskcla,args):

        super(Net,self).__init__()

        self.taskcla=taskcla
        self.args=args

        config = BertConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.bert = BertModel.from_pretrained(args.bert_model,config=config)
        self.FILTER_NUM=[100, 100, 100]

        self.efc1=torch.nn.Embedding(len(self.taskcla),args.bert_hidden_size)
        self.efc2=torch.nn.Embedding(len(self.taskcla),args.bert_hidden_size)
        self.ec1=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[0])
        self.ec2=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[1])
        self.ec3=torch.nn.Embedding(len(self.taskcla),self.FILTER_NUM[2])

        self.gate=torch.nn.Sigmoid()

        # self.relu=torch.nn.ReLU()

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.bert_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(args.bert_hidden_size,n))

        print('CONTEXTUAL + KIM + HAT')

        return

    def forward(self,t,input_ids, segment_ids, input_mask,s):

        output_dict = {}

        sequence_output, pooled_output = self.bert(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        masks=self.mask(t,s=s)

        #loss ==============
        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t,i in self.taskcla:
                y.append(self.last[t](pooled_output))

        output_dict['y'] = y
        output_dict['masks'] = masks
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict

    def mask(self,t,s=1):
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gfc1=self.gate(s*self.efc1(t))
        gfc2=self.gate(s*self.efc2(t))
        return [gc1,gc2,gc3,gfc1,gfc2]


    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks

        return None