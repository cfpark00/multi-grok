import utils
import argparse
import os
import shutil
import torch
import numpy as np
import time

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('yaml_path', type=str, help='Path to the yaml file')
    return parser.parse_args()

if __name__ == '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    config=utils.load_config(args.yaml_path)

    experiment_directory=config['experiment_directory']
    if os.path.exists(experiment_directory):
        shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory)
    shutil.copyfile(args.yaml_path,os.path.join(experiment_directory,'config.yaml'))

    seed=config['seed']
    utils.seed_all(seed)
    
    dataset_params=config['dataset_params']
    training_params=config['training_params']
    model_params=config['model_params']
    ###
    sampling_method=dataset_params['sampling_method']
    ##
    n_steps=training_params['n_steps']
    save_steps=training_params['save_steps']
    ckpt_steps=training_params['ckpt_steps']
    
    ############################
    dataset,tokenizer=utils.get_dataset_tokenizer(config)
    n_data,seq_len,vocab_size=len(dataset),dataset.get_seq_len(),tokenizer.get_vocab_size()

    model=utils.get_model(config,seq_len=seq_len,vocab_size=vocab_size)

    n_tr=int(dataset_params['train_frac']*n_data)
    batch_size=training_params['batch_size']
    n_val=n_data-n_tr
    ds_tr,ds_val=torch.utils.data.random_split(dataset,[n_tr,n_val])
    assert len(ds_tr)>batch_size and len(ds_val)>batch_size, "Batch size is too large for the dataset"
    if sampling_method is not None:
        tr_sampler=utils.get_sampler(sampling_method,ds_tr,tokenizer,**dataset_params["sampling_kwargs"])
        dl_tr=torch.utils.data.DataLoader(ds_tr,batch_size=batch_size,sampler=tr_sampler,drop_last=True)
        dl_val=torch.utils.data.DataLoader(ds_val,batch_size=batch_size,shuffle=True,drop_last=True)
    else:
        dl_tr=torch.utils.data.DataLoader(ds_tr,batch_size=batch_size,shuffle=True,drop_last=True)
        dl_val=torch.utils.data.DataLoader(ds_val,batch_size=batch_size,shuffle=True,drop_last=True)

    ### Training
    t=time.perf_counter()
    train_results=utils.train(model=model,tokenizer=tokenizer,dl_tr=dl_tr,dl_val=dl_val,device=device,save_steps=save_steps,ckpt_steps=ckpt_steps,n_steps=n_steps)
    t=time.perf_counter()-t
    print(f"Training took {t} seconds")
    time_tr=train_results['time_tr']
    time_val=train_results['time_val']
    t_loss=t-time_tr-time_val
    time_f_tr, time_f_val, time_f_loss=100*time_tr/t,100*time_val/t,100*t_loss/t
    print(f"Breakdown: Train:{time_f_tr:.2f}%, Val:{time_f_val:.2f}%, Loss:{time_f_loss:.2f}%")
    train_results['time_tot']=t

    ### Saving results
    ckpts_fol=os.path.join(experiment_directory,'ckpts')
    os.makedirs(ckpts_fol)
    ckpts=train_results['ckpts']
    for ckpt_step,ckpt in zip(ckpt_steps,ckpts):
        ckpt_path=os.path.join(ckpts_fol,f"ckpt_step={ckpt_step}.pth")
        torch.save(ckpt,ckpt_path)
    del train_results['ckpts']

    results_path=os.path.join(experiment_directory,'results.pth')
    torch.save(train_results,results_path)

    fig_path=os.path.join(experiment_directory,'results.png')
    utils.plot_results(train_results,fig_path)

