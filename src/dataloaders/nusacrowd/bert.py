#Coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import BertTokenizer as BertTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from nlp_data_utils import ABSATokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math

datasets = [
            './dat/nusacrowd/code_mixed_jv_id',
            './dat/nusacrowd/emot',
            './dat/nusacrowd/emotcmt',
            './dat/nusacrowd/imdb_jv',
            './dat/nusacrowd/karonese_sentiment',
            './dat/nusacrowd/smsa',

            './dat/nusacrowd/nusax_senti_ace',
            './dat/nusacrowd/nusax_senti_ban',
            './dat/nusacrowd/nusax_senti_bbc',
            './dat/nusacrowd/nusax_senti_bjn',
            './dat/nusacrowd/nusax_senti_bug',
            './dat/nusacrowd/nusax_senti_ind',
            './dat/nusacrowd/nusax_senti_jav',
            './dat/nusacrowd/nusax_senti_mad',
            './dat/nusacrowd/nusax_senti_min',
            './dat/nusacrowd/nusax_senti_nij',
            './dat/nusacrowd/nusax_senti_sun'
            ]


tasks = [
    'CodeMixedJVID_Javanese',
    'Emot_Indonesian',
    'EmotCMT_Indonesian',
    'IMDb_Javanese',
    'Sentiment_Karonese',
    'SmSA_Indonesian',
    
    'NusaX_Acehnese',
    'NusaX_Balinese',
    'NusaX_TobaBatak',
    'NusaX_Banjarese',
    'NusaX_Buginese', 
    'NusaX_Indonesian',
    'NusaX_Javanese',
    'NusaX_Madurese',
    'NusaX_Minangkabau',
    'NusaX_Ngaju',
    'NusaX_Sundanese'
    ]

def get(logger=None,args=None):
    data={}
    taskcla=[]

    # You can change the task heere
    f_name = 'nusacrowd_all_random'

    with open(f_name,'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()

    print('random_sep: ',random_sep)
    print('tasks: ', tasks)

    print('random_sep: ',len(random_sep))
    print('tasks: ',len(tasks))

    for t in range(args.ntasks):
        dataset = datasets[tasks.index(random_sep[t])]
        print('dataset: ',dataset)
        data[t]={}        
        data[t]['name']=dataset
        # data[t]['ncla']=8
        
        if 'IMDb_Javanese' == dataset:
            data[t]['ncla']=2
        elif ('Emot_Indonesian' == dataset) or ('EmotCMT_Indonesian' == dataset):
            data[t]['ncla']=5
        else:
            data[t]['ncla']=3

        processor = data_utils.NusaCrowdProcessor()
        label_list = processor.get_labels()
        tokenizer = ABSATokenizer.from_pretrained(args.bert_model)
        train_examples = processor.get_train_examples(dataset)

        num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs

        train_features = data_utils.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, "asc")
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps']=num_train_steps

        valid_examples = processor.get_dev_examples(dataset)
        valid_features=data_utils.convert_examples_to_features(
            valid_examples, label_list, args.max_seq_length, tokenizer, "asc")
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask, valid_all_label_ids,valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid']=valid_data


        processor = data_utils.NusaCrowdProcessor()
        label_list = processor.get_labels()
        tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        eval_examples = processor.get_test_examples(dataset)
        eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, "asc")

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids,all_tasks)
        # Run prediction for full data

        data[t]['test']=eval_data

        taskcla.append((t,int(data[t]['ncla'])))



    # Others
    n=0
    for t in data.keys():
        n+=data[t]['ncla']
    data['ncla']=n


    return data,taskcla