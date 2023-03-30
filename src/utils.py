import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def get_average(matrix):
    mat = np.array(matrix)

    result = []
    for i in range (len(mat)):
        result += [sum(mat[i])/(i+1)]

    return result

def get_filename(dir_name, exp_id, output, metrics):
  if metrics == "tasks":
    return f"{dir_name}{output}{metrics}.{exp_id}"

  return f"{dir_name}{output}progressive.{metrics}.{exp_id}"

def visualize(dir_name, exp_id, output, case_name, task):


  tasks_df = pd.read_csv(get_filename(dir_name, exp_id, output, "tasks"), sep="\s+", names=['Task'])
  if task == "nusacrowd":
    for i in range (len(tasks_df)):
        tasks_df['Task'][i] = f"{i+1}. {tasks_const[tasks_df['Task'][i]]}"

  # create visualization
  for metrics in list_metrics:
      if 'mtl' in output: # if model is MTL, only show last line
        try:
            if 'avg' in metrics:
                pass 
            else:
                df = pd.read_csv(get_filename(dir_name, exp_id, output, metrics), sep="\s+", names=[i for i in range (len(tasks_df['Task']))])
                df = df.drop(range(16))
                df.transpose().plot()
        except:
            print("File not found")
      
      else: # if model is not MTL, show all line
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
      plt.xlabel('number of tasks')
      plt.ylabel(metrics)
        
      if 'lss' not in metrics:
        plt.ylim(0, 1)
      
      plt.savefig(f"{output}_{metrics}_{exp_id}.png", bbox_inches='tight')
      
visualize('', 17, 'res/til_classification/nusacrowd/17 - bert_frozen_a-gem_.txt/bert_frozen_a-gem_.txt', 'nusacrowd_all_random', 'nusacrowd')