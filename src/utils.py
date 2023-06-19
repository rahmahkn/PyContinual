import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sb

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

########################################################################################################################

# for ACL
def report_tr(res, e, sbatch, clock0, clock1):
    # Training performance
    print(
        '| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train losses={:.3f} | T: loss={:.3f}, acc={:5.2f}% | D: loss={:.3f}, acc={:5.1f}%, '
        'Diff loss:{:.3f} |'.format(
            e + 1,
            1000 * sbatch * (clock1 - clock0) / res['size'],
            1000 * sbatch * (time.time() - clock1) / res['size'], res['loss_tot'],
            res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

def report_val(res):
    # Validation performance
    print(' Valid losses={:.3f} | T: loss={:.6f}, acc={:5.2f}%, | D: loss={:.3f}, acc={:5.2f}%, Diff loss={:.3f} |'.format(
        res['loss_tot'], res['loss_t'], res['acc_t'], res['loss_a'], res['acc_d'], res['loss_d']), end='')

########################################################################################################################



def fisher_matrix_diag_bert_ner(t,train,device,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets,valid_ids,label_mask, _= batch

        # Forward and backward
        model.zero_grad()
        outputs=model.forward(input_ids, segment_ids, input_mask,valid_ids,label_mask)

        loss=criterion(t,outputs[t],targets,label_mask)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher


def fisher_matrix_diag_ner_w2v(t,train,device,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        tokens_sentence_ids, targets,label_mask = batch

        # Forward and backward
        model.zero_grad()
        outputs=model.forward(tokens_sentence_ids,label_mask)

        loss=criterion(t,outputs[t],targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher



def fisher_matrix_diag_bert(t,train,device,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
#         b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)])))
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets,_= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(input_ids, segment_ids, input_mask)
        outputs = output_dict['y']

        loss=criterion(t,outputs[t],targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher



def fisher_matrix_diag_bert_dil(t,train,device,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets,_= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(input_ids, segment_ids, input_mask)
        output = output_dict['y']

        loss=criterion(t,output,targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

def fisher_matrix_diag_cnn(t,train,device,model,criterion,args,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        images,targets= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(images)
        if 'dil' in args.scenario:
            output = output_dict['y']
        elif 'til' in args.scenario:
            outputs = output_dict['y']
            output = outputs[t]

        loss=criterion(t,output,targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher


def fisher_matrix_diag_adapter_head(t,train,device,model,criterion,sbatch=20,
                                ce=None,lamb=None,mask_pre=None,args=None,ewc_lamb=None,model_old=None):
    # Init
    fisher={}
    for n,p in model.last.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        input_ids, segment_ids, input_mask, targets, _= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(t,input_ids, segment_ids, input_mask,s=args.smax)
        output = output_dict['y']
        masks = output_dict['masks']
        loss,reg=criterion(ce,lamb,mask_pre,output,targets,masks,
                            t=t,args=args,ewc_lamb=ewc_lamb,fisher=fisher,
                            model=model,model_old=model_old)

        loss.backward()
        # Get gradients
        for n,p in model.last.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.last.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher


def fisher_matrix_diag_cnn_head(t,train,device,model,criterion,sbatch=20,
                                ce=None,lamb=None,mask_pre=None,args=None,ewc_lamb=None,model_old=None):
    # Init
    fisher={}
    for n,p in model.last.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        images,targets= batch

        # Forward and backward
        model.zero_grad()
        task = torch.LongTensor([t]).cuda()
        output_dict=model.forward(task,images,s=args.smax)
        output = output_dict['y']
        masks = output_dict['masks']
        loss,reg=criterion(ce,lamb,mask_pre,output,targets,masks,
                            t=t,args=args,ewc_lamb=ewc_lamb,fisher=fisher,
                            model=model,model_old=model_old)

        loss.backward()
        # Get gradients
        for n,p in model.last.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.last.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

def fisher_matrix_diag_w2v(t,train,device,model,criterion,args,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()

    for i in tqdm(range(0,len(train),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,len(train)]))).cuda()
        batch=train[b]
        batch = [
            bat.to(device) if bat is not None else None for bat in batch]
        tokens_term_ids, tokens_sentence_ids, targets= batch

        # Forward and backward
        model.zero_grad()
        output_dict=model.forward(tokens_term_ids, tokens_sentence_ids)
        output = output_dict['y']
        if 'dil' in args.scenario:
            output = output_dict['y']
        elif 'til' in args.scenario:
            outputs = output_dict['y']
            output = outputs[t]
        loss=criterion(t,output,targets)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/len(train)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

def fisher_matrix_diag(t,x,y,model,criterion,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n]=0*p.data
    # Compute
    model.train()
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        images=torch.autograd.Variable(x[b],volatile=False)
        target=torch.autograd.Variable(y[b],volatile=False)
        # Forward and backward
        model.zero_grad()
        outputs=model.forward(images)
        loss=criterion(t,outputs[t],target)
        loss.backward()
        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

########################################################################################################################

# define metrics
list_metrics = ['acc', 'f1_macro', 'lss', 'avg_acc', 'avg_f1_macro', 'avg_lss']

# define tasks
tasks_const = {
'./dat/nusacrowd/code_mixed_jv_id': 'CodeMixedJVID_Javanese',
'./dat/nusacrowd/emot': 'Emot_Indonesian',
'./dat/nusacrowd/emotcmt': 'EmotCMT_Indonesian',
'./dat/nusacrowd/imdb_jv': 'IMDb_Javanese',
'./dat/nusacrowd/karonese_sentiment': 'Sentiment_Karonese',
'./dat/nusacrowd/smsa': 'SmSA_Indonesian',
'./dat/nusacrowd/nusax_senti_ace': 'NusaX_Acehnese',
'./dat/nusacrowd/nusax_senti_ban': 'NusaX_Balinese',
'./dat/nusacrowd/nusax_senti_bbc': 'NusaX_TobaBatak',
'./dat/nusacrowd/nusax_senti_bjn': 'NusaX_Banjarese',
'./dat/nusacrowd/nusax_senti_bug': 'NusaX_Buginese',
'./dat/nusacrowd/nusax_senti_ind': 'NusaX_Indonesian',
'./dat/nusacrowd/nusax_senti_jav': 'NusaX_Javanese',
'./dat/nusacrowd/nusax_senti_mad': 'NusaX_Madurese',
'./dat/nusacrowd/nusax_senti_min': 'NusaX_Minangkabau',
'./dat/nusacrowd/nusax_senti_nij': 'NusaX_Ngaju',
'./dat/nusacrowd/nusax_senti_sun': 'NusaX_Sundanese'    
}

tasks_id_const = {
'./dat/nusacrowd/code_mixed_jv_id': 0,
'./dat/nusacrowd/emot': 1,
'./dat/nusacrowd/emotcmt': 2,
'./dat/nusacrowd/imdb_jv': 3,
'./dat/nusacrowd/karonese_sentiment': 4,
'./dat/nusacrowd/smsa': 5,
'./dat/nusacrowd/nusax_senti_ace': 6,
'./dat/nusacrowd/nusax_senti_ban': 7,
'./dat/nusacrowd/nusax_senti_bbc': 8,
'./dat/nusacrowd/nusax_senti_bjn': 9,
'./dat/nusacrowd/nusax_senti_bug': 10,
'./dat/nusacrowd/nusax_senti_ind': 11,
'./dat/nusacrowd/nusax_senti_jav': 12,
'./dat/nusacrowd/nusax_senti_mad': 13,
'./dat/nusacrowd/nusax_senti_min': 14,
'./dat/nusacrowd/nusax_senti_nij': 15,
'./dat/nusacrowd/nusax_senti_sun': 16
}

transform_setting = {
    "bert": "BERT",
    "bert_frozen": "BERT (Frozen)",
    "bert_adapter": "BERT + Adapter",
    
    "a-gem": "A-GEM",
    "ewc": "EWC",
    "hat": "HAT",
    "kan": "KAN",
    "one": "MONO",
    "mtl": "MULTI",
    "ncl": "VANILLA",
    "ctr": "CTR",
    "b-cl": "B-CL"
}

def transform_list(list):
    return [transform_setting[elmt] for elmt in list]

def get_average(matrix):
    mat = np.array(matrix)
    n_tasks = len(mat)

    result = []
    for i in range (n_tasks):
        result += [sum(mat[i][0:i+1]) / (i+1)]

    return result

def get_filename(dir_name, exp_id, output, metrics):
  if metrics == "tasks":
    return f"{dir_name}{output}{metrics}.{exp_id}"

  return f"{dir_name}{output}progressive.{metrics}.{exp_id}"

def get_output(backbone, baseline, exp_id):
    return f"res/til_classification/nusacrowd/{exp_id} - {backbone}_{baseline}_.txt/{backbone}_{baseline}_.txt"

def get_one_result(mat):
    return [mat[i][i] for i in range (len(mat))]
    
def calculate(dataframe, type):
  if type == "avg":
    mean = dataframe.mean(axis=1)
    return [mean[i] for i in range(len(dataframe))]

  elif type == "std":
    std = dataframe.std(axis=1)
    return [std[i] for i in range(len(dataframe))]

def calculate_avg_metrics(mat):
  avg_per_iter = []

  for row in mat:
    avg_per_iter += [np.average(row)]

  return "{:.4f}".format(np.average(avg_per_iter))

def calculate_bwt(mat):
  sigma = 0
  n = len(mat) # number of tasks

  for i in range(0, n-1):
    sigma += (mat[n-1][i] - mat[i][i])

  return "{:.4f}".format(sigma / (n-1))

def calculate_fwt(mat, b):
  sigma = 0
  n = len(mat) # number of tasks

  for i in range(1, n):
    sigma += (mat[i-1][i] - b[i])

  return "{:.4f}".format(sigma / (n-1))

def calculate_metrics(data):
    result = []
    
    for index, row in data:
        # load data from text
        exp_id = row['exp_id']
        baseline = row['baseline']
        backbone = row['backbone']
        
        output = get_output(backbone, baseline, exp_id)
        filename = f'res/til_classification/nusacrowd/b_vector/{backbone}_{baseline}_.txtprogressive.b.{exp_id}'
        
        with open(f'{output}progressive.acc.{exp_id}') as fp:
            mat_acc = [list(map(float, line.strip().split('\t'))) for line in fp]
            
        with open(f'{output}progressive.f1_macro.{exp_id}') as fp:
            mat_f1_macro = [list(map(float, line.strip().split('\t'))) for line in fp]
            
        with open(f'{output}progressive.lss.{exp_id}') as fp:
            mat_lss = [list(map(float, line.strip().split('\t'))) for line in fp]
            
        # with open(filename) as fp:
        #     vec_b = [list(map(float, line.strip().split('\t'))) for line in fp][0]
        
        # calculate result    
        if baseline == 'one':
            result.append([exp_id, backbone, baseline,
                calculate_avg_metrics([get_one_result(mat_acc)]), calculate_avg_metrics([get_one_result(mat_f1_macro)]), calculate_avg_metrics([get_one_result(mat_lss)]),
                calculate_avg_metrics([get_one_result(mat_acc)]), calculate_avg_metrics([get_one_result(mat_f1_macro)]), calculate_avg_metrics([get_one_result(mat_lss)]),
                0, 0])
        elif baseline == 'mtl':
            result.append([exp_id, backbone, baseline,
                calculate_avg_metrics([mat_acc[-1]]), calculate_avg_metrics([mat_f1_macro[-1]]), calculate_avg_metrics([mat_lss[-1]]),
                calculate_avg_metrics([mat_acc[-1]]), calculate_avg_metrics([mat_f1_macro[-1]]), calculate_avg_metrics([mat_lss[-1]]),
                0, 0])
        else:
            result.append([exp_id, backbone, baseline,
                calculate_avg_metrics(get_average(mat_acc)), calculate_avg_metrics(get_average(mat_f1_macro)), calculate_avg_metrics(get_average(mat_lss)),
                calculate_avg_metrics([mat_acc[-1]]), calculate_avg_metrics([mat_f1_macro[-1]]), calculate_avg_metrics([mat_lss[-1]]),
                calculate_bwt(mat_acc), 0])
    
    # change result to dataframe to sort
    result_df = pd.DataFrame(result, columns=['exp_id', 'backbone', 'baseline', 'avg_acc', 'avg_f1_macro', 'avg_lss', 'last_acc', 'last_f1', 'lss', 'bwt', 'fwt'])
    result_df = result_df.astype(dtype= {'avg_acc': 'float64', 'avg_f1_macro': 'float64', 'avg_lss': 'float64', 'last_acc': 'float64', 'last_f1': 'float64', 'lss': 'float64', 'bwt': 'float64', 'fwt': 'float64'})
    result_df = result_df.sort_values(by='last_f1', ascending=False)
    
    result_df_aggr = result_df.groupby(['backbone', 'baseline']).aggregate({'last_acc': 'mean', 'last_f1': 'mean', 'bwt': 'mean', 'fwt': 'mean'})
    result_df_aggr = result_df_aggr.sort_values(by='last_f1', ascending=False)
    
    print(result_df_aggr)
    
    # write result to csv file    
    with open('res/til_classification/result.csv', 'a', newline='') as fp:
        csv_writer = csv.writer(fp, delimiter=',')
        
        for row in result_df.values.tolist():
            csv_writer.writerow(row)
            
    # with open('res/til_classification/result_per_setting.csv', 'a', newline='') as fp:
    #     csv_writer = csv.writer(fp, delimiter=',')
        
    #     list_row = result_df_aggr.values.tolist()
    #     for i in range (len(list_row)):
    #         csv_writer.writerow([result_df_aggr.index.get_level_values(0).to_list()[i]] + [result_df_aggr.index.get_level_values(1).to_list()[i]] + ['{:,.4f}'.format(elmt) for elmt in list_row[i]])

def get_worst_forgetting(dir_name, exp_id, backbone, baseline, list_task): # n: jumlah data worst yang dipakai
    output = get_output(backbone, baseline, exp_id)
    df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics="f1_macro"), sep="\s+", names=[i for i in range (17)])
    mat = df.values.tolist()
    
    n_task = len(mat)
    worst = {'delta': 0, 'id_task_effected': 0, 'id_task_effecting': 0}
    
    for i in range (n_task-1):
        for j in range (0, i+1):
            if mat[i+1][j]-mat[i][j] < worst['delta']:
                worst['delta'] = mat[i+1][j]-mat[i][j]
                worst['id_task_effected'] = j
                worst['id_task_effecting'] = i+1

    # print(worst['id_task_effected'])
    worst['delta'] = "%.4f" % worst['delta']
    worst['name_task_effected'] = list_task[worst['id_task_effected']]
    worst['name_task_effecting'] = list_task[worst['id_task_effecting']]
                
    return worst

def get_best_transfer(dir_name, exp_id, backbone, baseline, list_task): # n: jumlah data worst yang dipakai
    output = get_output(backbone, baseline, exp_id)
    df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics="f1_macro"), sep="\s+", names=[i for i in range (17)])
    mat = df.values.tolist()
    
    n_task = len(mat)
    worst = {'delta': 0, 'id_task_effected': 0, 'id_task_effecting': 0}
    
    for i in range (n_task-1):
        for j in range (0, i+1):
            if mat[i+1][j]-mat[i][j] > worst['delta']:
                worst['delta'] = mat[i+1][j]-mat[i][j]
                worst['id_task_effected'] = j
                worst['id_task_effecting'] = i+1

    # print(worst['id_task_effected'])
    worst['delta'] = "%.4f" % worst['delta']
    worst['name_task_effected'] = list_task[worst['id_task_effected']]
    worst['name_task_effecting'] = list_task[worst['id_task_effecting']]
                
    return worst

########################################################################################################################

def visualize(dir_name, exp_id, output, case_name, task):
  tasks_df = pd.read_csv(get_filename(dir_name, exp_id, output, "tasks"), sep="\s+", names=['Task'])
  if task == "nusacrowd":
    for i in range (len(tasks_df)):
        tasks_df['Task'][i] = f"{i+1}. {tasks_const[tasks_df['Task'][i]]}"

  # create visualization
  for metrics in list_metrics:
      if 'mtl' in output: # if model is MTL, only show last line
        try:
            # if 'avg' in metrics:
            #     # base_metrics = metrics.replace('avg_', '')
            df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics.replace('avg_', '')), sep="\s+", names=[i for i in range (len(tasks_df['Task']))])
            df = df.drop(range(16))
            df.transpose().plot()
        except:
            print("File not found")
            
      elif 'one' in output: # if model is ONE, take the diagonal value
        try:
            n = len(tasks_df['Task'])
            
            base_metrics = metrics.replace('avg_', '')
            df = pd.read_csv(get_filename(dir_name, exp_id, output, base_metrics), sep="\s+", names=[i for i in range (n)])
            
            for i in df.index:
                df.at[n-1, i] = np.mean([df.at[j, i] for j in range(i+1)])
            
            df = df.drop(range(16))
            
            if 'avg' in metrics:
                np.savetxt(output + f'progressive.{metrics}.' + str(exp_id), df,'%.4f',delimiter='\t')
            
            df.transpose().plot()
        except:
            print("File not found")
      
      else: # if model is not MTL/ONE, show all line
        try:
            if 'avg' in metrics:
                base_metrics = metrics.replace('avg_', '')
                base_df = pd.read_csv(get_filename(dir_name, exp_id, output, base_metrics), sep="\s+", names=[i for i in range (len(tasks_df['Task']))])
                np.savetxt(output + f'progressive.{metrics}.' + str(exp_id), get_average(base_df),'%.4f',delimiter='\t')
                df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics), sep="\s+", names=[i for i in range (len(tasks_df['Task']))])

            else:
                df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics), sep="\s+", names=[i for i in range (len(tasks_df['Task']))])
        except:
            print("File not found")
        
        df.plot()

      title = output.split('/')[-1]
      plt.legend(tasks_df['Task'], bbox_to_anchor=(1.0, 1.0))
      plt.title(f'{title.replace("_.txt", "")} - {case_name}')
      plt.xticks(range(17), [i for i in range (1, 18)])
      plt.xlabel('task')
      plt.ylabel(metrics)
        
      if 'lss' not in metrics:
        plt.ylim(0, 1)
      
      plt.savefig(f"{output}_{metrics}_{exp_id}.png", bbox_inches='tight')

def create_viz(list_dataframe, title, legend, xlabel, ylabel, filename, list_exp_id):    
    color_const = {
        'A-GEM': ['c-', 'c'],
        'EWC': ['m-', 'm'],
        'HAT': ['y-', 'y'],
        'KAN': ['g-', 'g'],
        'CTR': ['g-', 'g'],
        'B-CL': ['k-', 'k'],
        'VANILLA': ['r-', 'r'],
        
        'MONO': ['b--', 'b'],
        'MULTI': ['k--', 'k'],
        
        'BERT': ['c-', 'c'],
        'BERT (Frozen)': ['m-', 'm'],
        'BERT + Adapter': ['y-', 'y']
    }
    
    
    for i in range (len(list_dataframe)):
            x_indices = [(i+1) for i in range(len(list_dataframe[i]))]
            avg = calculate(list_dataframe[i], "avg")
            std = calculate(list_dataframe[i], "std")
            
            plt.plot(x_indices, avg, color_const[legend[i]][0], label=legend[i])
            plt.fill_between(x_indices, [(avg[i]-std[i]) for i in range (len(avg))], [(avg[i]+std[i]) for i in range (len(avg))], color= color_const[legend[i]][1], alpha=0.05)
    
    plt.title(title.replace('_.txt', ''))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    # if 'lss' not in filename:
    #     plt.ylim(0, 1)

    plt.savefig(filename, bbox_inches='tight')
    plt.close()
        
def merge_viz(dir_name, list_exp_id, list_backbone, list_baseline, case_name, metrics, typ):
  list_df = []
  
  for i in range (len(list_exp_id)):
    n = len(list_exp_id[i])

    # bikin n dataframe
    df = pd.DataFrame(columns=list_exp_id[i])

    for exp_id in list_exp_id[i]:
        if typ == 'multi_backbone':
            baseline = list_baseline[0]
            backbone = list_backbone[i]
            output = get_output(backbone, baseline, exp_id)
        else: # type == 'multi_baseline'
            baseline = list_baseline[i]
            backbone = list_backbone[0]
            output = get_output(backbone, baseline, exp_id)
    
        if baseline == "mtl":
            df_mtl = pd.read_csv(get_filename(dir_name, exp_id, output, metrics.replace('avg_', '')), sep="\s+", names=[i for i in range(17)])
            # print(get_average(df_mtl))
            df[exp_id] = get_average(df_mtl)
        elif baseline == "one":
            df_one = pd.read_csv(get_filename(dir_name, exp_id, output, metrics), sep="\s+", names=[i for i in range(17)])
            # print(exp_id, df_one)
            df[exp_id] = df_one.iloc[0].values.tolist()
            # print(df_one.iloc[0].values.tolist())
        else:
            df[exp_id] = pd.read_csv(get_filename(dir_name, exp_id, output, metrics), sep="\s+", names=[exp_id])
        
    list_df.append(df)
    
  list_backbone = transform_list(list_backbone)
  list_baseline = transform_list(list_baseline)

  if typ == 'multi_backbone':
      create_viz(list_df, f'{baseline} - {case_name}', list_backbone, 'task', metrics, f"viz/{baseline}/{baseline}_{metrics}.png", list_exp_id)
  else: # type == 'multi_baseline'
      create_viz(list_df, f'{backbone} - {case_name}', list_baseline, 'task', metrics, f"viz/{backbone}/{backbone}_{metrics}.png", list_exp_id)

def run_create_viz(list_backbone, list_baseline, title, typ):
    list_exp = pd.read_csv('res/til_classification/list_experiments.csv',delimiter=',')
    
    list_exp_id = []
    
    if typ == 'multi_backbone':
        for backbone in list_backbone:
            list_exp_id.append(list_exp[(list_exp['backbone'] == backbone) & (list_exp['baseline'] == list_baseline[0])]['exp_id'].to_list())
    else:
        for baseline in list_baseline:
            list_exp_id.append(list_exp[(list_exp['backbone'] == list_backbone[0]) & (list_exp['baseline'] == baseline)]['exp_id'].to_list())
    
    for metrics in ['avg_acc', 'avg_f1_macro', 'avg_lss']:
        merge_viz('', list_exp_id, list_backbone, list_baseline, title, metrics, typ)

########################################################################################################################

def create_transfer(exp_id, output, metric):
    # import file with data
    data = pd.read_csv(get_filename('', exp_id, output, metric),  sep="\t", names=[i for i in range(17)])
    data_list = data.values.tolist()
    
    df_transfer = [data_list[0]]
    
    for i in range (1, 17):
        df_transfer.append([(data_list[i][j]-data_list[i-1][j]) for j in range(17)])
    
    np.savetxt(output + f'progressive.transfer.{metric}.' + str(exp_id), df_transfer, '%.4f', delimiter='\t')

def create_heatmap(exp_id, backbone, baseline, metric):
    output = get_output(backbone, baseline, exp_id)
    
    # import task name
    tasks_df = pd.read_csv(get_filename('', exp_id, output, "tasks"), sep="\s+", names=['Task'])
    for i in range (len(tasks_df)):
        tasks_df['Task'][i] = tasks_const[tasks_df['Task'][i]]
    
    # import file with data
    data = pd.read_csv(get_filename('', exp_id, output, metric),  sep="\t", names=tasks_df['Task'].values)
    data['Task'] = tasks_df['Task'].values
    data.set_index('Task', inplace=True)
        
    plt.figure(figsize=(12, 8))
    
    # plotting correlation heatmap
    dataplot = sb.heatmap(data)
    
    # saving heatmap
    plt.subplots_adjust(bottom=0.3, left=0.4)
    plt.title(f"{exp_id} - {backbone}_{baseline}")
    plt.savefig(f"{output}_{metric}_{exp_id}.png", bbox_inches='tight')
    
def merge_heatmap(list_exp_id, backbone, baseline):
    # get list of dataframe
    list_df = []
    for exp_id in list_exp_id:
        # get task name
        output = get_output(backbone, baseline, exp_id)
        tasks_df = pd.read_csv(get_filename('', exp_id, output, "tasks"), sep="\s+", names=['Task'])
        
        # make dataframe for each exp
        df = pd.read_csv(get_filename('', exp_id, output, 'transfer.f1_macro'), sep="\s+", names=[tasks_id_const[tasks_df['Task'].values.tolist()[i]] for i in range(17)])
        list_df.append(df)
    
    result_df = pd.DataFrame(columns=[i for i in range(17)])
    for i in range(17):
        result_df[i] = [np.average([list_df[0][i][j], list_df[1][i][j], list_df[2][i][j]]) for j in range(17)]
    
    result_df.set_axis(tasks_const.values(), axis='columns', inplace=True)
    result_df['Task'] = tasks_const.values()
    result_df.set_index('Task', inplace=True)
    np.savetxt(f"viz/heatmap/{backbone}_{baseline}_f1_macro", result_df, '%.4f', delimiter='\t')
    
    # create heatmap
    plt.figure(figsize=(12, 8))
    
    # plotting correlation heatmap
    dataplot = sb.heatmap(result_df)
    
    # saving heatmap
    plt.subplots_adjust(bottom=0.3, left=0.4)
    plt.title(f"heatmap - {backbone}_{baseline}")
    plt.savefig(f"viz/heatmap/{backbone}_{baseline}_f1_macro.png", bbox_inches='tight')
    plt.close()

########################################################################################################################
            
if __name__ == "__main__":
    list_exp = pd.read_csv('res/til_classification/list_experiments.csv',delimiter=',')
    
    with open('nusacrowd_all_random', 'r') as f:
        list_task = [[task.strip() for task in line.split(' ')] for line in f]
    
    # visualize an experiment
    # for index, row in list_exp.iterrows():
    #     if (row['baseline'] == 'hat') and (row['backbone'] == 'bert_adapter'):
    #         visualize('', row['exp_id'], f"res/til_classification/nusacrowd/{row['exp_id']} - {row['backbone']}_{row['baseline']}_.txt/{row['backbone']}_{row['baseline']}_.txt", 'nusacrowd_all_random', 'nusacrowd')
    
    # create viz for backbone and baseline combination
    # list_create_viz = [
    #     # multi_baseline
    #     [['bert'], ['mtl', 'one', 'ncl', 'ewc', 'a-gem'], 'multi_baseline'],
    #     [['bert_adapter'], ['mtl', 'hat', 'a-gem', 'ewc', 'b-cl', 'ctr'], 'multi_baseline'],
    #     [['bert_frozen'], ['one', 'a-gem', 'hat', 'kan', 'ncl', 'ewc'], 'multi_baseline'],
        
    #     # multi_backbone
    #     [['bert_adapter', 'bert_frozen', 'bert'], ['a-gem'], 'multi_backbone'],
    #     [['bert_adapter', 'bert_frozen', 'bert'], ['ewc'], 'multi_backbone'],
    #     [['bert_adapter', 'bert_frozen'], ['hat'], 'multi_backbone'],
    #     [['bert_frozen'], ['kan'], 'multi_backbone'],
    #     [['bert', 'bert_adapter'], ['mtl'], 'multi_backbone'],
    #     [['bert', 'bert_frozen'], ['ncl'], 'multi_backbone'],
    #     [['bert', 'bert_frozen'], ['one'], 'multi_backbone']
    # ]
    
    # for elmt in list_create_viz:
    #     run_create_viz(elmt[0], elmt[1], 'nusacrowd all random', elmt[2])
    
    # recalculate an experiment
    # calculate_metrics(81, 'bert_adapter', 'a-gem')
    
    # recalculate all experiments        
    calculate_metrics(list_exp.iterrows())
    
    # get worst forgetting
    # with open('res/til_classification/result_transfer_cl.csv', 'a', newline='') as fp:
    #     for index, elmt in list_exp.iterrows():
    #         csv_writer = csv.writer(fp, delimiter=',')
            
    #         result = get_best_transfer('', elmt['exp_id'], elmt['backbone'], elmt['baseline'], list_task[elmt['id_random']])
    #         csv_writer.writerow([elmt['exp_id'], elmt['backbone'], elmt['baseline'], result['delta'], result['name_task_effected'], result['name_task_effecting'], result['id_task_effected'], result['id_task_effecting']])
    
    # create transfer matrix
    # for index, row in list_exp.iterrows():
    #     if (row['baseline'] != 'mtl') or (row['baseline'] != 'one'):
    #         create_transfer(row['exp_id'], get_output(row['backbone'], row['baseline'], row['exp_id']), 'f1_macro')
    
    # # create heatmap
    # create_heatmap(111, 'bert_adapter', 'b-cl', 'transfer.f1_macro')
    # for index, row in list_exp.iterrows():
    #     if (row['baseline'] != 'mtl') or (row['baseline'] != 'one'):
    #         create_heatmap(row['exp_id'], row['backbone'], row['baseline'], 'transfer.f1_macro')
    
    # create merge heatmap
    # list_setting = [
    #     ['bert', 'mtl'],
    #     ['bert', 'one'],
    #     ['bert', 'ncl'],
    #     ['bert_adapter', 'mtl'],
    #     ['bert_adapter', 'hat'],
    #     ['bert_adapter', 'ewc'],
    #     ['bert_adapter', 'a-gem'],
    #     ['bert_frozen', 'one'],
    #     ['bert_frozen', 'ewc'],
    #     ['bert_frozen', 'a-gem'],
    #     ['bert_frozen', 'ncl'],
    #     ['bert_frozen', 'hat'],
    #     ['bert_frozen', 'kan'],
    #     ['bert_adapter', 'b-cl']
    # ]
    
    # for setting in list_setting:
    #     list_exp_id = []
    #     for index, row in list_exp.iterrows():
    #         if (row['baseline'] == setting[1]) and (row['backbone'] == setting[0]):
    #             list_exp_id.append(row['exp_id'])
                
    #     merge_heatmap(list_exp_id, setting[0], setting[1])