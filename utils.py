import numpy as np
import torch
import time
import tqdm
import yaml
yaml.Dumper.ignore_aliases = lambda *args : True
import copy
import matplotlib.pyplot as plt
import os

import gpt_model

task_names_=["add","sub","max","first","rand","even","a2b","a2abb2","a3ab"]

def get_dataset(task_name,**kwargs):
    if task_name in task_names_:
        return globals()[f"get_{task_name}_dataset"](**kwargs)
    else:
        raise ValueError("Task not supported")

def get_add_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a+b)%n_max
    strdata=[f"{a[i]},add,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_sub_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a-b)%n_max
    strdata=[f"{a[i]},sub,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_max_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=np.maximum(a,b)
    strdata=[f"{a[i]},max,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_first_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=a
    strdata=[f"{a[i]},first,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_rand_dataset(n_max=113,seed=0):
    np.random.seed(seed)
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=np.random.randint(0,n_max,len(a))
    strdata=[f"{a[i]},rand,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_even_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a+b)%2
    strdata=[f"{a[i]},even,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_a2b_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a**2+b)%n_max
    strdata=[f"{a[i]},a2b,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_a2abb2_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a**2+a*b+b**2)%n_max
    strdata=[f"{a[i]},a2abb2,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

def get_a3ab_dataset(n_max=113):
    nums=np.arange(0,n_max)
    a,b=np.meshgrid(nums,nums,indexing='ij')
    a,b=a.reshape(-1),b.reshape(-1)
    c=(a**3+a*b)%n_max
    strdata=[f"{a[i]},a3ab,{b[i]},=,{c[i]},<EOS>" for i in range(len(a))]
    return strdata

class Tokenizer():
    def __init__(self,n_max=113):
        self.n_max=n_max
        self.encoder={}
        for i in range(n_max):
            self.encoder[str(i)]=i
        n_curr=n_max
        self.encoder["="]=n_curr
        n_curr+=1
        self.encoder["<EOS>"]=n_curr
        n_curr+=1
        for task_name in task_names_:
            self.encoder[task_name]=n_curr
            n_curr+=1

        self.decoder={v:k for k,v in self.encoder.items()}

    def encode_(self,el):
        if el in self.encoder:
            return self.encoder[el]
        else:
            assert False, f"Unknown token {el}"

    def encode(self,seq):
        out=[]
        for el in seq:
            out.append(self.encode_(el))
        return out
    
    def decode_el(self,el):
        el=int(el)
        if el in self.decoder:
            return self.decoder[el]
        else:
            assert False, f"Unknown token {el}"

    def decode(self,seq):
        out=[]
        for el in seq:
            out.append(self.decode_el(el))
        return out

    def get_vocab_size(self):
        return len(self.encoder)
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self,file_paths,tokenizer):
        self.seqs=[]
        for file_path in file_paths:
            with open(file_path,"r") as f:
                for line in f:
                    self.seqs.append(line.strip().split(","))
        self.tokenizer=tokenizer
        self.tokenized=[]
        for seq in self.seqs:
            self.tokenized.append(self.tokenizer.encode(seq))
        self.tokenized=torch.tensor(self.tokenized)

    def __len__(self):
        return self.tokenized.shape[0]

    def get_seq_len(self):
        return self.tokenized.shape[-1]

    def __getitem__(self,idx):
        return self.tokenized[idx]
    
def get_task_names_corrects(logits,labels,tokenizer):
    #both logits and labels indice i represent token i+1
    i_tasks=labels[:,0]# this contains the task name
    task_names=tokenizer.decode(i_tasks)

    pred4=torch.argmax(logits[:,3,:],dim=-1)#indice=4 is predicted by indice=3
    gt4=labels[:,3]
    corrects=(pred4==gt4).detach().cpu().numpy()
    return task_names,corrects

def train(model,tokenizer,dl_tr,dl_val,device,save_steps,ckpt_steps,n_steps):
    online_losses=[]
    online_accs=[]
    train_losses=[]
    train_hits_counts=[]
    val_losses=[]
    val_hits_counts=[]
    ckpts=[]
    step=0

    model.to(device)
    pbar=tqdm.tqdm(total=n_steps,desc="Training")
    time_tr=0
    time_val=0
    timer=time.perf_counter()
    while True:
        model.train()
        for batch in dl_tr:
            batch=batch.to(device)
            in_tokens=batch[:,:-1]
            target_tokens=batch[:,1:]
            logits,loss=model(in_tokens,target_tokens)
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            model.optimizer.step()

            online_losses.append(loss.item())
            task_names,corrects=get_task_names_corrects(logits,target_tokens,tokenizer)
            online_accs.append(corrects.astype(float).mean())
            step+=1
            pbar.update()
            if step in save_steps:
                time_tr+=time.perf_counter()-timer
                timer=time.perf_counter()
                model.eval()
                with torch.no_grad():
                    train_loss=0
                    train_hits={task_name:0 for task_name in task_names_}
                    train_counts={task_name:0 for task_name in task_names_}
                    for batch in dl_tr:
                        batch=batch.to(device)
                        in_tokens=batch[:,:-1]
                        target_tokens=batch[:,1:]
                        logits,loss=model(in_tokens,target_tokens)
                        train_loss+=loss.item()
                        task_names,corrects=get_task_names_corrects(logits,target_tokens,tokenizer)
                        for task_name,correct in zip(task_names,corrects):
                            train_hits[task_name]+=correct
                            train_counts[task_name]+=1
                    train_loss/=len(dl_tr)
                    train_losses.append(train_loss)
                    train_hits_counts.append({"hits":train_hits,"counts":train_counts})

                    val_loss=0
                    val_hits={task_name:0 for task_name in task_names_}
                    val_counts={task_name:0 for task_name in task_names_}
                    for batch in dl_val:
                        batch=batch.to(device)
                        in_tokens=batch[:,:-1]
                        target_tokens=batch[:,1:]
                        logits,loss=model(in_tokens,target_tokens)
                        val_loss+=loss.item()
                        task_names,corrects=get_task_names_corrects(logits,target_tokens,tokenizer)
                        for task_name,correct in zip(task_names,corrects):
                            val_hits[task_name]+=correct
                            val_counts[task_name]+=1
                    val_loss/=len(dl_tr)
                    val_losses.append(val_loss)
                    val_hits_counts.append({"hits":val_hits,"counts":val_counts})
                model.train()
                time_val+=time.perf_counter()-timer
                timer=time.perf_counter()
            if step in ckpt_steps:
                ckpts.append(copy.deepcopy(model.state_dict()))
            if step==n_steps:
                break
        if step==n_steps:
            break
    pbar.close()


    train_accs={task_name:[] for task_name in task_names_}
    val_accs={task_name:[] for task_name in task_names_}
    for i in range(len(save_steps)):
        train_hits,train_counts=train_hits_counts[i]["hits"],train_hits_counts[i]["counts"]
        val_hits,val_counts=val_hits_counts[i]["hits"],val_hits_counts[i]["counts"]
        for task_name in task_names_:
            train_accs[task_name].append(np.divide(train_hits[task_name],train_counts[task_name],where=train_counts[task_name]>0))
            val_accs[task_name].append(np.divide(val_hits[task_name],val_counts[task_name],where=val_counts[task_name]>0))

    return_dict={
        "save_steps":save_steps,
        "online_losses":online_losses,
        "online_accs":online_accs,

        "train_losses":train_losses,
        "train_hits_counts":train_hits_counts,
        "train_accs":train_accs,

        "val_losses":val_losses,
        "val_hits_counts":val_hits_counts,
        "val_accs":val_accs,

        "ckpts":ckpts,
        "time_tr":time_tr,
        "time_val":time_val
    }
    return return_dict

default_config={
    "experiment_directory":"./data/experiments/default/",
    "seed":0,
    "dataset_params":{
        "base_path":"./data/datasets/",
        "n_max":53,
        "seq_len":6,
        "task_names":["even","add","rand"],
        "train_frac":0.5,
        "sampling_method":"random",
    },
    "training_params":{
        "n_steps":100_000,
        "batch_size":256,
        "save_steps":[10,100,1_000,10_000,100_000],
        "ckpt_steps":[],
    },
    "model_params":{
        "lr":0.001,
        "weight_decay":1.0,
        "betas":[0.9,0.98],
        "network_params":{
            "n_layer":1,
            "n_head":4,
            "n_embd":128,
            "dropout":0.0,
            "bias":False
            }
    }
}

def check_config(config):
    #required
    assert "experiment_directory" in config, "experiment_directory not in config"
    assert "seed" in config, "seed not in config"
    assert "dataset_params" in config, "dataset_params not in config"
    assert "training_params" in config, "training_params not in config"
    assert "model_params" in config, "model_params not in config"
    assert "base_path" in config["dataset_params"], "base_path not in dataset_params"
    assert "n_max" in config["dataset_params"], "n_max not in dataset_params"
    assert "task_names" in config["dataset_params"], "task_names not in dataset_params"
    assert "train_frac" in config["dataset_params"], "train_frac not in dataset_params"
    assert "sampling_method" in config["dataset_params"], "sampling_method not in dataset_params"
    assert "n_steps" in config["training_params"], "n_steps not in training_params"
    assert "batch_size" in config["training_params"], "batch_size not in training_params"
    assert "save_steps" in config["training_params"], "save_steps not in training_params"
    assert "lr" in config["model_params"], "lr not in model_params"
    assert "weight_decay" in config["model_params"], "weight_decay not in model_params"
    assert "betas" in config["model_params"], "betas not in model_params"
    assert "network_params" in config["model_params"], "network_params not in model_params"
    assert "n_layer" in config["model_params"]["network_params"], "n_layer not in network_params"
    assert "n_head" in config["model_params"]["network_params"], "n_head not in network_params"
    assert "n_embd" in config["model_params"]["network_params"], "n_embd not in network_params"
    assert "dropout" in config["model_params"]["network_params"], "dropout not in network_params"
    assert "bias" in config["model_params"]["network_params"], "bias not in network_params"


def write_config(config,file_path):
    check_config(config)
    yaml.dump(config,open(file_path,"w"))

def load_config(file_path):
    config=yaml.load(open(file_path,"r"),Loader=yaml.Loader)
    check_config(config)
    return config

def get_dataset_tokenizer(config):
    dataset_params=config["dataset_params"]
    n_max=dataset_params["n_max"]
    tokenizer=Tokenizer(n_max=n_max)
    task_names=dataset_params['task_names']
    file_paths=[]
    for task_name in task_names:
        file_paths.append(os.path.join(dataset_params["base_path"],f"n_max={n_max}",task_name+".txt"))
    dataset=Dataset(file_paths=file_paths,tokenizer=tokenizer)#pre-tokenizing dataset
    return dataset,tokenizer    

def get_sampler(sampling_method,dataset,tokenizer,**kwargs):
    if sampling_method=="random":
        return torch.utils.data.RandomSampler(dataset)
    elif sampling_method=="add_first":
        assert "temp" in kwargs, "temp not in kwargs"
        temp=kwargs["temp"]
        t_add=1.
        t_base=0.
        logits=[]
        for i in range(len(dataset)):
            if dataset[i][1]==tokenizer.encode_("add"):
                logits.append(t_add)
            else:
                logits.append(t_base)
        assert np.unique(logits).shape[0]==2
        logits=torch.tensor(logits)
        probs=torch.nn.functional.softmax(logits/temp,dim=0)
        return torch.utils.data.WeightedRandomSampler(probs,len(probs),replacement=True)
    elif sampling_method=="add_last":
        assert "temp" in kwargs, "temp not in kwargs"
        temp=kwargs["temp"]
        t_add=0.
        t_base=1.
        logits=[]
        for i in range(len(dataset)):
            if dataset[i][1]==tokenizer.encode_("add"):
                logits.append(t_add)
            else:
                logits.append(t_base)
        assert np.unique(logits).shape[0]==2
        logits=torch.tensor(logits)
        probs=torch.nn.functional.softmax(logits/temp,dim=0)
        return torch.utils.data.WeightedRandomSampler(probs,len(probs),replacement=True)
    else:
        raise ValueError("Sampling method not supported")

def get_model(config,seq_len,vocab_size):
    model_params=config["model_params"]
    network_params=model_params["network_params"]
    gpt_config=gpt_model.GPTConfig(block_size=seq_len,vocab_size=vocab_size,**network_params)
    model=gpt_model.GPT(gpt_config)
    if "ckpt_path" in model_params:
        model.load_state_dict(torch.load(model_params["ckpt_path"]))
        print("Loaded model from checkpoint")
    model.optimizer=torch.optim.AdamW(model.parameters(),lr=model_params["lr"],weight_decay=model_params["weight_decay"],betas=model_params["betas"])
    return model

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)

def plot_results(results,save_path=None):
    save_steps=results["save_steps"]
    online_losses=results["online_losses"]
    train_losses=results["train_losses"]
    val_losses=results["val_losses"]
    online_accs=results["online_accs"]
    train_hits_counts=results["train_hits_counts"]
    train_accs=results["train_accs"]
    val_hits_counts=results["val_hits_counts"]
    val_accs=results["val_accs"]


    fig=plt.figure(figsize=(20,10))
    plt.rcParams.update({'font.size': 15})
    plt.subplot(1,2,1)
    plt.plot(online_losses,ls="--",c="black",alpha=0.5,label="Online Train Loss")
    plt.plot(save_steps,train_losses,ls="--",c="blue",label="Train Set Loss")
    plt.plot(save_steps,val_losses,ls="-",c="red",label="Val. Set Loss")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(left=10)
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(online_accs,ls="--",c="black",alpha=0.5,label="Online Acc.")
    for task_name in task_names_:
        if val_hits_counts[0]["counts"][task_name]==0:
            continue
        plt.plot(save_steps,train_accs[task_name],ls="--",label="Task: "+task_name+" Train Acc.")
        plt.plot(save_steps,val_accs[task_name],c=plt.gca().lines[-1].get_color(),label="Task: "+task_name+" Val. Acc.")
    plt.xscale("log")
    plt.xlim(left=10)
    plt.ylim(-0.1,1.1)
    plt.yticks(np.linspace(0,1,11))
    plt.ylabel("Accuracy")
    plt.xlabel("Steps")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        return fig