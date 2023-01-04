import numpy as np
import pandas as pd
import yaml
from sklearn import metrics
from easydict import EasyDict as edict
import os
from sklearn.metrics import confusion_matrix

## Functions for calculating reliability
def cal_minAP(n_pos, n_neg):
    # theoretical minimum average precision
    AP_min = 0.0
    for i in range(1, n_pos + 1):
        AP_min += i / (i + n_neg)
    AP_min = AP_min / n_pos
    return AP_min

def cal_mis_pm(labels, score):
    #  calculate the misclassification performance measures
    #  include AUROC, AP, nAP
    #  labels: labels for postive or not
    #  scores: quantified uncertainty; dict
    n_sample = len(labels)  # the total number of predictions
    n_pos = np.sum(labels)  # the total number of positives
    n_neg = n_sample - n_pos  # the total number of negatives
    skew = n_pos / n_sample
    #AUROC = {key: [] for key in scores}

    if skew == 0:  # No postive samples found
        #  PR curve makes no sense, record it but dont use it
        AP= float("nan")  # AP
        nAP = float("nan")  # normalised AP
    else:
        minAP = cal_minAP(n_pos, n_neg)

        AUROC = metrics.roc_auc_score(labels, score)
        AP = metrics.average_precision_score(labels, score)
        #  normalised AP
        nAP = (AP - minAP) / (1 - minAP)
    return AUROC, AP, nAP

# Function to update reliability
def update_reliability(df_base_selected, cfg):

    for un in cfg.RESULTS.reliability.scores:
        #un="un_nentropy"
        #un="un_nnmp"
        #un="un_vac"
        #un="un_diss"
        if df_base_selected[un].isnull().values.any():
            continue

        AUROC_list = []
        AUPR_list = []
        nAUPR_list = []
        for sb_n in cfg.TEST_CONFIG.sb_list:
            df_tmp = df_base_selected[(df_base_selected["sb"]==sb_n)&(df_base_selected["state"]=='test')]
            labels = df_tmp['actual']!=df_tmp['predict']
            AUROC, AUPR, nAUPR = cal_mis_pm(labels, df_tmp[un])
            AUROC_list.append(AUROC)
            AUPR_list.append(AUPR)
            nAUPR_list.append(nAUPR)


        n = len(AUROC_list)
        R_dict={
            'run': [cfg.run]*n,
            #'distilled': [cfg.distilled]*n,
            'model': [cfg.model]*n,
            'sb': cfg.TEST_CONFIG.sb_list,
            'score': [un]*n,
            'AUROC': AUROC_list,
            'AUPR': AUPR_list,
            'nAUPR': nAUPR_list
        }

        df_new = pd.DataFrame(R_dict)
        file_path = cfg.RESULT_PATH.analysed+cfg.RESULTS.reliability.filename
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, dtype=cfg.RESULTS.reliability.DATA_TYPE)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new

        df.to_csv(file_path, float_format=np.float16, index=False)


def check_and_find_threshold(f1_scores, thresholds, df_tmp_sb, un):

    f1_scores_sorted = np.sort(f1_scores)
    threshold = thresholds[np.argmax(f1_scores)]
    for i in range(1,len(f1_scores_sorted)+1):
        threshold = thresholds[np.argwhere(f1_scores==f1_scores_sorted[-i])[0][0]]
        for day_n, time_n in ([1,1], [1,2],[2,1],[2,2]):
            df_tmp = df_tmp_sb[(df_tmp_sb['day']==day_n)&(df_tmp_sb["time"]==time_n)]
            valid_pred =  df_tmp[un] < threshold
            if np.sum(valid_pred)<round(0.1*len(valid_pred)):
                break
        if day_n==2 and time_n==2:
            break
    return threshold

def update_rej_CM_multi_dimension(cfg):

    cm = np.zeros([8,8,9]) 
    beta=2
    for run_i in range(1,11):
        df_base_selected =  pd.read_csv(f'{cfg.RESULT_PATH.loaded}run{run_i}_{cfg.model}.csv', dtype=cfg.DATA_TYPE)

        for n, sb_n in enumerate(cfg.TEST_CONFIG.sb_list):
            df_tmp_sb=df_base_selected[df_base_selected["sb"]==sb_n]
            df_tmp = df_tmp_sb[df_tmp_sb["state"]=='val']
            Y_val = df_tmp['actual']==df_tmp['predict']
            precision, recall, thresholds = metrics.precision_recall_curve(~Y_val, df_tmp["un_vac"])
            f1_scores = (1+beta**2)*recall*precision/(beta**2*precision+recall+1e-07)
            threshold_vac = check_and_find_threshold(f1_scores, thresholds, df_tmp, "un_vac")
        
            precision, recall, thresholds = metrics.precision_recall_curve(~Y_val, df_tmp["un_diss"])
            f1_scores = (1+beta**2)*recall*precision/(beta**2*precision+recall+1e-07)
            threshold_diss = check_and_find_threshold(f1_scores, thresholds, df_tmp_sb, "un_diss")

            df_tmp_test = df_tmp_sb[df_tmp_sb["state"]=='test']

            Y_test = df_tmp_test['actual']==df_tmp_test['predict']
            valid_pred =  (df_tmp_test["un_vac"] < threshold_vac) & (df_tmp_test["un_diss"] < threshold_diss)
            if np.sum(valid_pred)!=0:
                cm[n,...,0:-1] += confusion_matrix(df_tmp_test[valid_pred]['actual'],df_tmp_test[valid_pred]['predict'])
        
            for i in range(8):
                n_i = np.sum(df_tmp_test['actual']==i)
                n_valid_i = np.sum(df_tmp_test[valid_pred]['actual']==i)
                n_rej_i = n_i-n_valid_i
                cm[n,i,-1]+=n_rej_i

    np.save(cfg.RESULT_PATH.analysed+f'{cfg.model}_multi_rej_cm.npy',cm)

def update_CM_baseline(cfg):

    cm = np.zeros([8,8,8])
    for run_i in range(1,11):
        df_base_selected =  pd.read_csv(f'{cfg.RESULT_PATH.loaded}run{run_i}_{cfg.model}.csv', dtype=cfg.DATA_TYPE)
        for n, sb_n in enumerate(cfg.TEST_CONFIG.sb_list):
            df_tmp_sb=df_base_selected[df_base_selected["sb"]==sb_n]
            df_tmp_test = df_tmp_sb[df_tmp_sb["state"]=='test']
            Y_test = df_tmp_test['actual']==df_tmp_test['predict']
            cm[n] += confusion_matrix(df_tmp_test['actual'],df_tmp_test['predict'])
    
    np.save(cfg.RESULT_PATH.analysed+f'{cfg.model}_cm.npy',cm)

def update_rej_R_multi_dimension(df_base_selected, cfg):

    TAR_list = []
    TRR_list = []
    RR_list = []
    threshold_vac_list = []
    threshold_diss_list = []
    sb_list=[]
    day_list = []
    period_list = []
    
    beta=2
    for sb_n in cfg.TEST_CONFIG.sb_list:
        df_tmp_sb=df_base_selected[df_base_selected["sb"]==sb_n]
            #un="un_vac"
            #un="un_diss"
        df_tmp = df_tmp_sb[df_tmp_sb["state"]=='val']
        Y_val = df_tmp['actual']==df_tmp['predict']
        precision, recall, thresholds = metrics.precision_recall_curve(~Y_val, df_tmp["un_vac"])
        f1_scores = (1+beta**2)*recall*precision/(beta**2*precision+recall+1e-07)
        threshold_vac = check_and_find_threshold(f1_scores, thresholds, df_tmp, "un_vac")
        
        precision, recall, thresholds = metrics.precision_recall_curve(~Y_val, df_tmp["un_diss"])
        f1_scores = (1+beta**2)*recall*precision/(beta**2*precision+recall+1e-07)
        threshold_diss = check_and_find_threshold(f1_scores, thresholds, df_tmp_sb, "un_diss")


        for day_n in cfg.TEST_CONFIG.day_list[1]:
            df_tmp_day=df_tmp_sb[df_tmp_sb['day']==day_n]
            for time_n in cfg.TEST_CONFIG.period_list:
                period ='AM' if time_n==1 else 'PM'
                df_tmp_time = df_tmp_day[df_tmp_day["time"]==time_n]
                Y_test = df_tmp_time['actual']==df_tmp_time['predict']
                n_total = len(Y_test)
                valid_pred =  (df_tmp_time["un_vac"] < threshold_vac) & (df_tmp_time["un_diss"] < threshold_diss)
                #valid_pred =  (df_tmp_time["un_vac"] < threshold_vac) | (df_tmp_time["un_diss"] < threshold_diss)
                TA_label = Y_test[valid_pred]
                n_accepted = len(TA_label)
                n_rejected = n_total-n_accepted
                TA = np.sum(TA_label) # True positive(accepted)
                TR = np.sum(Y_test) - TA # True negative(rejected)
                FA = n_accepted - TA # False positive(accepted)
                FR = n_rejected - TR # False negative(rejected)
                TRR = np.nan if n_rejected==0 else FR/n_rejected
                TAR = np.nan if n_accepted ==0 else TA/n_accepted
                RR = n_rejected/n_total
                period_list.append(period)
                TAR_list.append(TAR)
                TRR_list.append(TRR)
                RR_list.append(RR)
                day_list.append(day_n)
                sb_list.append(sb_n)
                threshold_vac_list.append(threshold_vac)
                threshold_diss_list.append(threshold_diss)
    n = len(TAR_list)
    rej_dict={
        'run': [cfg.run]*n,
        'model': [cfg.model]*n,
        'sb': sb_list,
        'day': day_list,
        'period': period_list,
        'threshold_vac': threshold_vac_list,
        'threshold_diss': threshold_diss_list,
        'TAR': TAR_list,
        'TRR': TRR_list,
        'RR': RR_list
    }

    df_new = pd.DataFrame(rej_dict)
    file_path = cfg.RESULT_PATH.analysed+cfg.RESULTS.rejection_multi.filename
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype=cfg.RESULTS.rejection_multi.DATA_TYPE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(file_path, float_format=np.float16, index=False)


# Function to update accuracy
def update_rej_R(df_base_selected, cfg):

    TAR_list = []
    TRR_list = []
    RR_list = []
    threshold_list = []
    sb_list=[]
    score_list = []
    day_list = []
    period_list = []
    #J_list = [] # Youden's J statistic

    for sb_n in cfg.TEST_CONFIG.sb_list:
        df_tmp_sb=df_base_selected[df_base_selected["sb"]==sb_n]
        for un in cfg.RESULTS.reliability.scores:
            #un="un_nentropy"
            #un="un_nnmp"
            #un="un_vac"
            #un="un_diss"
            if df_base_selected[un].isnull().values.any():
                continue
            df_tmp = df_tmp_sb[df_tmp_sb["state"]=='val']
            Y_test = df_tmp['actual']==df_tmp['predict']
            #precision, recall, thresholds = metrics.precision_recall_curve(Y_test, 1-df_tmp[un])
            precision, recall, thresholds = metrics.precision_recall_curve(~Y_test, df_tmp[un])
            beta=2
            f1_scores = (1+beta**2)*recall*precision/(beta**2*precision+recall+1e-07)
            #threshold = thresholds[np.argmax(f1_scores)]
            threshold = check_and_find_threshold(f1_scores, thresholds, df_tmp_sb, un)
            for day_n in cfg.TEST_CONFIG.day_list[1]:
                df_tmp_day=df_tmp_sb[df_tmp_sb['day']==day_n]
                for time_n in cfg.TEST_CONFIG.period_list:
                    period ='AM' if time_n==1 else 'PM'
                    df_tmp_time = df_tmp_day[df_tmp_day["time"]==time_n]
                    Y_test = df_tmp_time['actual']==df_tmp_time['predict']
                    n_total = len(Y_test)
                    valid_pred =  df_tmp_time[un] < threshold
                    TA_label = Y_test[valid_pred]
                    n_accepted = len(TA_label)
                    n_rejected = n_total-n_accepted
                    TA = np.sum(TA_label) # True positive(accepted)
                    TR = np.sum(Y_test) - TA # True negative(rejected)
                    FA = n_accepted - TA # False positive(accepted)
                    FR = n_rejected - TR # False negative(rejected)
                    TRR = 0.0 if n_rejected==0 else FR/n_rejected
                    TAR = 0.0 if n_accepted ==0 else TA/n_accepted
                    RR = n_rejected/n_total
                    #sensitivity = 0.0 if TA+FR==0 else TA/(TA+FR)
                    #specificity = 0.0 if TR+FA==0 else TR/(TR+FA)
                    #J = sensitivity + specificity - 1
                    period_list.append(period)
                    TAR_list.append(TAR)
                    TRR_list.append(TRR)
                    RR_list.append(RR)
                    day_list.append(day_n)
                    sb_list.append(sb_n)
                    threshold_list.append(threshold)
                    score_list.append(un)
                    #J_list.append(J)

    n = len(TAR_list)
    acc_dict={
        'run': [cfg.run]*n,
        #'distilled': [cfg.distilled]*n,
        'model': [cfg.model]*n,
        'sb': sb_list,
        'day': day_list,
        'period': period_list,
        'threshold': threshold_list,
        'score': score_list,
        'TAR': TAR_list,
        'TRR': TRR_list,
        'RR': RR_list
        #'J': J_list
    }

    df_new = pd.DataFrame(acc_dict)
    file_path = cfg.RESULT_PATH.analysed+cfg.RESULTS.rejection.filename
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype=cfg.RESULTS.rejection.DATA_TYPE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(file_path, float_format=np.float16, index=False)

# Function to update accuracy
def update_acc(df_base_selected, cfg):

    acc_list = []
    sb_list=[]
    day_list = []
    period_list = []
    state_list = []
    for sb_n in cfg.TEST_CONFIG.sb_list:
        df_tmp_sb=df_base_selected[df_base_selected["sb"]==sb_n]
        for day_n in [1,2,3,4,5]:
            state = "test" if day_n in cfg.TEST_CONFIG.day_list[1] else "val"
            df_tmp = df_tmp_sb[df_tmp_sb["state"]==state]
            df_tmp_day=df_tmp[df_tmp['day']==day_n]
            for time_n in cfg.TEST_CONFIG.period_list:
                period ='AM' if time_n==1 else 'PM'
                df_tmp_time = df_tmp_day[df_tmp_day["time"]==time_n]
                Y_test = df_tmp_time['actual']==df_tmp_time['predict']
                n_total = len(Y_test)
                acc = np.sum(Y_test)/n_total
                period_list.append(period)
                acc_list.append(acc)
                day_list.append(day_n)
                sb_list.append(sb_n)
                state_list.append(state)

    n = len(acc_list)
    acc_dict={
        'run': [cfg.run]*n,
        #'distilled': [cfg.distilled]*n,
        'model': [cfg.model]*n,
        'sb': sb_list,
        'state': state_list,
        'day': day_list,
        'period': period_list,
        'acc': acc_list
    }

    df_new = pd.DataFrame(acc_dict)
    file_path = cfg.RESULT_PATH.analysed+cfg.RESULTS.accuracy.filename
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, dtype=cfg.RESULTS.accuracy.DATA_TYPE)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(file_path, float_format=np.float16, index=False)


if __name__ == "__main__":
    with open("results_analyse.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
    for key in cfg.DATA_TYPE:
        cfg.DATA_TYPE[key] = eval(cfg.DATA_TYPE[key])

    for key in cfg.RESULTS.reliability.DATA_TYPE:
        cfg.RESULTS.reliability.DATA_TYPE[key] = eval(cfg.RESULTS.reliability.DATA_TYPE[key])

    for key in cfg.RESULTS.accuracy.DATA_TYPE:
        cfg.RESULTS.accuracy.DATA_TYPE[key] = eval(cfg.RESULTS.accuracy.DATA_TYPE[key])

    for key in cfg.RESULTS.rejection.DATA_TYPE:
        cfg.RESULTS.rejection.DATA_TYPE[key] = eval(cfg.RESULTS.rejection.DATA_TYPE[key])
   
    for key in cfg.RESULTS.rejection_multi.DATA_TYPE:
        cfg.RESULTS.rejection_multi.DATA_TYPE[key] = eval(cfg.RESULTS.rejection_multi.DATA_TYPE[key])

    cfg.RESULT_PATH.loaded = os.getcwd()+cfg.RESULT_PATH.loaded
    cfg.RESULT_PATH.analysed = os.getcwd()+cfg.RESULT_PATH.analysed
    #cfg.distilled=True
    '''
    for i in range(1,11):
        cfg.run=i
        for model in ["ecnn0", "ecnn1", "ecnn2", "ecnn3"]:
            df_base_selected =  pd.read_csv(f'{cfg.RESULT_PATH.loaded}run{i}_{model}.csv', dtype=cfg.DATA_TYPE)
            cfg.model = model
            update_acc(df_base_selected, cfg)
            update_rej_R(df_base_selected, cfg)
    

    
    for i in range(1,11):
        cfg.run=i
        for model in ["ecnn1", "ecnn2", "ecnn3"]:
            df_base_selected = pd.read_csv(f'{cfg.RESULT_PATH.loaded}run{i}_{model}.csv', dtype=cfg.DATA_TYPE)
            cfg.model = model
            update_rej_R_multi_dimension(df_base_selected, cfg)
    '''
    for model in ["ecnn1", "ecnn2"]:
        cfg.model=model
        update_rej_CM_multi_dimension(cfg)
        #update_CM_baseline(cfg)

    #cfg.model="ecnn1"
    #update_rej_CM_multi_dimension(cfg)

