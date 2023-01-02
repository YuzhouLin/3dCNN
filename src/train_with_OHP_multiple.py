import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
import yaml
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, WeightedRandomSampler
from scipy.stats import hmean
#from torchsummary import summary
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl (annealing); \
        3: edl with kl (trade-off)')

args = parser.parse_args()

EDL_USED = args.edl
DEVICE = pre.try_gpu()

def run_training(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    n_class = len(cfg.CLASS_NAMES)
    if cfg.TRAINING.day_n==1:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy')), axis=0)
    elif cfg.TRAINING.day_n==2:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t2.npy')), axis=0)

    
    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32))# ([5101, 1, 14, 400])
    #print(X_train_torch.size())
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))


    W_train_torch = torch.ones(len(Y_train_torch),dtype=torch.float32)
    W_val_torch = torch.ones(len(Y_val_torch),dtype=torch.float32)

    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch, W_val_torch)

    _, train_class_counts = np.unique(train_Y, return_counts=True)
    _, val_class_counts = np.unique(val_Y, return_counts=True)

    print(train_class_counts)
    print(val_class_counts)
    n_train = train_class_counts.sum()
    n_val = val_class_counts.sum()
    class_weights_train = [float(n_train)/train_class_counts[i] for i in range(n_class)]
    class_weights_val = [float(n_val)/val_class_counts[i] for i in range(n_class)]

    weights_train = train_Y
    weights_val = val_Y
    for i in range(n_class):
        weights_train[train_Y==i] = class_weights_train[i]
        weights_val[val_Y==i] = class_weights_val[i]
    sampler_train = WeightedRandomSampler(weights_train, int(n_train),replacement=True)
    sampler_val = WeightedRandomSampler(weights_val, int(n_val),replacement=True)
    # load_data
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.HP.batch_size, sampler=sampler_train, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.HP.batch_size, sampler=sampler_val, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)

    trainloaders = {
        "train": train_loader,
        "val": val_loader,
    }


    for run_i in range(1,11): # train many times for validation purpose
        # Load Model
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)
    
        if not cfg.TRAINING.retrained_from_scratch:
            print('train from best_hpo')
            checkpoint = torch.load(cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt')#, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])


        model.to(DEVICE)
        #summary(model, input_size=(1,400,2,7), batch_size=-1)
        optimizer = getattr(
            torch.optim,cfg.HP.optimizer)(model.parameters(), lr=cfg.HP.lr, weight_decay=cfg.HP.weight_decay, betas=(0.5, 0.999))

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.HP.lr_factor, patience=int(cfg.TRAINING.early_stopping_iter/2), verbose=True, eps=cfg.HP.scheduler_eps)
        eng = utils.EngineTrain(model, optimizer, device=DEVICE)

        loss_params = {'edl_used': EDL_USED, 'device': DEVICE}
        if EDL_USED != 0:
            loss_params['class_n'] = n_class
            for item in cfg.HP_SEARCH[f'EDL{EDL_USED}']:
                loss_params[item]=cfg.HP[item]

        best_loss = np.inf
        early_stopping_iter = cfg.TRAINING.early_stopping_iter

        torch.backends.cudnn.benchmark = True
        for epoch in range(1, cfg.TRAINING.epochs + 1):
            # if 'annealing_step' in loss_params:
            if EDL_USED == 2:
                loss_params['epoch_num'] = epoch
            train_losses, tmp_pred, tmp_true = eng.train(trainloaders, loss_params)
            train_loss = train_losses['train']
            valid_loss = train_losses['val']
            scheduler.step(valid_loss)

            print(
                f"epoch:{epoch}, "
                f"train_loss: {train_loss}, "
                f"valid_loss: {valid_loss}. "
            )
            if valid_loss < best_loss:
                best_loss = valid_loss
                early_stopping_counter = 0

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                    }, cfg.model_path+f'/{cfg.TRAINING.retrained_model_name}_sb{sb_n}_run{run_i}.pt')
            else:
                early_stopping_counter += 1
            if early_stopping_counter > early_stopping_iter:
                break

    return

def prepared_cfg(sb_n):

    # Load config file
    with open("hpo_search.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    #sb_n=cfg.DATA_CONFIG.sb_n
    cfg.DATA_CONFIG.sb_n = sb_n
    # Check study path
    study_dir = f'ecnn{EDL_USED}'
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
        hp_study = yaml.load(f, Loader=yaml.SafeLoader)

    cfg.best_loss = hp_study[0]

    for key, item in hp_study[1].items():
        cfg.HP[key] =  item
    cfg.HP['batch_size'] = cfg.HP['batch_base']*2**cfg.HP['batch_factor']

    # Check model saved path
    cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)

    # Write (Save) HP to a yaml file

    return cfg


if __name__ == "__main__":

    #print(time.ctime())
    for sb_n in [3,4,5]:
        cfg = prepared_cfg(sb_n)
        run_training(cfg)
    #print(time.ctime())

    os.system('shutdown')
