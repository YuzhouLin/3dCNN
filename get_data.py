import scipy.io as io
import numpy as np
import os
import yaml
import pickle as pkl


DATA_READ='../../hy-tmp/Data6/Processed/'
DATA_SAVE='../../hy-tmp/Data6/'

def seg_emg(samples, wl=400, ratio_non_overlap=0.1):
    segmented = []
    for n in range(0, samples.shape[0]-wl, round(wl*ratio_non_overlap)):
        segdata = samples[n:n+wl,:] # 400*14
        segmented.append(np.expand_dims(segdata, axis=0))
    return segmented

def data_prepared(cfg):
    class_list = [0, 1, 3, 4, 6, 9, 10, 11]

    sb_n = cfg.sb_n
    train_folder = DATA_SAVE+f'{sb_n}/train/'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    val_folder = DATA_SAVE+f'{sb_n}/val/'
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    test_folder = DATA_SAVE+f'{sb_n}/test/'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # prepare train and val data
    for day_n in cfg.DATA_CONFIG.day_list:
	for time_n in cfg.DATA_CONFIG.time_list:
	    train_X = np.array([])
            train_Y = np.array([])
	    val_X = np.array([])
            val_Y = np.array([])
	    for k in range(1, cfg.DATA_CONFIG.trial_n):
    		with open(f"{DATA_READ}S{sb_n}_D{day_n}_T{time_n}_t{k}.pkl", "rb") as f:
        	    data = pkl.load(f)
	        if k in cfg.DATA_CONFIG.trial_val:
		    val_X = np.concatenate((data.X, data.Y), axis=0) ## need to check data.X 
                    val_Y = np.concatenate((data.X, data.Y), axis=0)
		else:                
	            train_X = np.concatenate((data.X, data.Y), axis=0)
		    train_Y = np.concatenate((data.X, data.Y), axis=0)

	    np.save(train_folder+f'X_d{day_n}_t{time_n}.npy', np.array(train_X, dtype=np.float32))
	    np.save(train_folder+f'Y_d{day_n}_t{time_n}.npy', np.array(train_Y, dtype=np.int64))
	    np.save(val_folder+f'X_d{day_n}_t{time_n}.npy', np.array(val_X, dtype=np.float32))
	    np.save(val_folder+f'Y_d{day_n}_t{time_n}.npy', np.array(val_Y, dtype=np.int64))
    


    # prepare test data
    for day_n in cfg.DATA_CONFIG.day_test_list:
        for time_n in time_list:
 	    test_X = np.array([])
            test_Y = np.array([])
	    for k in range(1, cfg.DATA_CONFIG.trial_n):
    		with open(f"{DATA_READ}S{sb_n}_D{day_n}_T{time_n}_t{k}.pkl", "rb") as f:
        	    data = pkl.load(f)
	        
	        test_X = np.concatenate((data.X, data.Y), axis=0)
		test_Y = np.concatenate((data.X, data.Y), axis=0)
            np.save(test_folder+f'/X_d{day_n}_t{time_n}.npy', np.array(test_X, dtype=np.float32))
            np.save(test_folder+f'/Y_d{day_n}_t{time_n}.npy', np.array(test_Y, dtype=np.int64))


if __name__ == "__main__":
    # Load config file
    with open("hpo_search.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    for sb_n in [1]:#[1,3,4,5,6,7,8,10]:
        cfg.sb_n = sb_n
	data_prepared(cfg)

