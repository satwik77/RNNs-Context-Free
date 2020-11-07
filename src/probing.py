import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import pickle
from attrdict import AttrDict
from src.model import LanguageModel, build_model, train_model, run_validation
from src.utils.helper import *
from src.utils.logger import get_logger
from src.utils.sentence_processing import sents_to_idx
from src.utils.dyck_generator import DyckLanguage
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb
import argparse
import csv

def_run_name = 'DEPTH-EXPv2-generalizeTrue-modeltypeRNN-no-debugTrue-hiddensize16-optrmsprop-vallowerwindow52-testsize5000-lr0005-datasetDyck-2-LengthwiseBins-modetrain-valupperwindow100-trainingsize10000-epochs100-depth2'

def init_setup(run_name, log_folder, device):
    model_path = os.path.join('models', run_name)

    #Load Vocabulary
    voc_file = os.path.join(model_path, 'vocab.p')
    with open(voc_file, 'rb') as f:
        voc = pickle.load(f)
        
    #Load config file
    config_file = os.path.join(model_path, 'config.p')
    with open(config_file, 'rb') as f:
        config = AttrDict(pickle.load(f))
    config.pos_encode_type = 'absolute'
    config.max_period = 10000
        
    #Initialize logger
    config.log_path = os.path.join(log_folder, run_name)
    config.bias = False
    config.extraffn = False
    if os.path.exists(config.log_path) == False:
        os.mkdir(config.log_path)
    log_file = os.path.join(config.log_path, 'log.text')
    logger = get_logger('visualize', log_file, logging.DEBUG)
    
    #Build model
    model = build_model(config=config, voc=voc, device = device, logger=logger)
    
    #Load checkpoint
    checkpoint = get_latest_checkpoint(model_path, logger)
    try:
        ep_offset, train_loss, score, _ = load_checkpoint(model, config.mode, checkpoint, logger, device, bins = config.bins)
    except:
        ep_offset, train_loss, score, _ = load_checkpoint(model, config.mode, checkpoint, logger, device, bins = 5)
    return model, voc, config, logger

def get_depths(seqs, dlang, group = False):
    depths = []
    for seq in seqs:
        depth = dlang.depth_counter(seq).sum(1).astype(int)
        if group:
            depths.append(depth)
        else:
            depths += list(depth)
    
    return np.array(depths)
        
    
def get_seqs_n_depths(num_par, path = 'data/Dyck-2-LengthwiseBins/'):
    dlang = DyckLanguage(p=0.5, q=0.25, num_pairs = num_par)

    with open(os.path.join(path, 'train_src.txt')) as f:
        train_seqs = f.read().split('\n')

        if train_seqs[-1] == '':
            train_seqs = train_seqs[:-1]

    with open(os.path.join(path, 'val_src_bin0.txt')) as f:
        val_seqs = f.read().split('\n')
        if val_seqs[-1] == '':
            val_seqs = val_seqs[:-1]

    val_seqs = list(set(val_seqs) - set(train_seqs))
    train_depths = get_depths(train_seqs, dlang)
    val_depths = get_depths(val_seqs, dlang)
    val_depths_grouped = get_depths(val_seqs, dlang, group = True)
    
    return train_seqs, val_seqs, train_depths, val_depths, val_depths_grouped

def get_random_labels(seqs, labels_dict = {}, l = 0, h = 18):
    all_labels = []
    all_labels_grouped = []
    for seq in seqs:
        labels = []
        for i in range(len(seq)):
            subseq = seq[:i+1]
            if subseq in labels_dict:
                labels.append(labels_dict[subseq])
            else:
                label = np.random.randint(l, h+1)
                labels.append(label)
                labels_dict[subseq] = label
        all_labels += labels
        all_labels_grouped.append(np.array(labels).astype(int))
            
    return np.array(all_labels).astype(int), np.array(all_labels_grouped), labels_dict

def get_cell_states(model, seqs, voc, layer = -1, group = False, hidden_id = 1, device = 'cuda:0'):

    cell_states = []
    for i,seq in enumerate(seqs):
        with torch.no_grad():
            seq_tensor = sents_to_idx(voc, [seq]).transpose(0,1).to(device)[:-1]
            out, hiddens = model.model(seq_tensor, None, lengths = [len(seq_tensor)], get_step_wise_info= True)
            if group:
                c = torch.cat([hidden[hidden_id][layer] for hidden in hiddens]).detach().cpu().numpy()
                cell_states.append(c)
            else:
                c = list(torch.cat([hidden[hidden_id][layer] for hidden in hiddens]).detach().cpu().numpy())
                cell_states += c
            print("Completed {}/{}".format(i+1, len(seqs)), end = '\r', flush = True)
    cell_states = np.array(cell_states)
    print()
    
    return cell_states

def full_seq_accuracy(X_grouped, y_grouped, probe):
    num_correct = 0
    for (X, y) in zip(X_grouped, y_grouped):
        preds = probe.predict(X)
        num_correct += int((preds == y).all())
    accuracy = num_correct / len(X_grouped)
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_par', type = int, default = 2, help = "Dyck-n, n= num_par")
    parser.add_argument('-dataset', default = 'Dyck-2-LengthwiseBins', help = 'Dataset containing Dyck-n strings')
    parser.add_argument('-run_name', default = def_run_name, help = "To identify the folder containing the pretrained models")
    parser.add_argument('-layer', type = int, default = 0, help = "Layer of rnn to use to get the cell states")
    parser.add_argument('-mlp_hid', type = int, default = 32, help = "Hidden size of MLP probe")
    parser.add_argument('-hidden_id', type = int, default = 0, choices = [0,1], help = "0 for hidden states and 1 for cell state")
    parser.add_argument('-gpu', type = int, default = 0, help = "GPU id")
    args = parser.parse_args()

    device = gpu_init_pytorch(args.gpu)
    log_folder = 'logs'
    run_name = args.run_name
    rep_type = 'hidden_state' if args.hidden_id == 0 else 'cell_state'
    num_par = args.num_par

    print("Loading Pretrained Model...")
    model, voc, config, logger = init_setup(run_name, log_folder, device)
    print(model.model)

    logger.info("Loading Data...")
    train_seqs, val_seqs, train_depths, val_depths, val_depths_grouped = get_seqs_n_depths(num_par = num_par, path= os.path.join('data', args.dataset))

    logger.info("Number of Training Sequences: {}".format(len(train_seqs)))
    logger.info("Number of Training Substrings: {}".format(len(train_depths)))
    logger.info("Number of Validation Sequences: {}".format(len(val_seqs)))
    logger.info("Number of Validation Substrings: {}".format(len(val_depths))) 
    logger.info("Depth range: [{},{}]".format(train_depths.min(), train_depths.max()))

    logger.info("Computing the cell states")
    train_cell_states = get_cell_states(model, train_seqs, voc, layer = args.layer, hidden_id= args.hidden_id, device = device)
    val_cell_states = get_cell_states(model, val_seqs, voc, layer = args.layer, hidden_id= args.hidden_id, device = device)
    val_cell_states_grouped = get_cell_states(model, val_seqs, voc, group = True, layer = args.layer, hidden_id= args.hidden_id, device = device)

    np.random.seed(0)
    train_random_labels,_, labels_dict = get_random_labels(train_seqs, l = train_depths.min(), h = train_depths.max())
    val_random_labels,val_random_labels_grouped, _ = get_random_labels(val_seqs, labels_dict)

    logger.info("Training a Linear Probe on Legit Data")
    linear_probe = LogisticRegression(multi_class = 'multinomial', random_state=42, solver='lbfgs')
    linear_probe.fit(train_cell_states, train_depths)
    linear_probe_train_acc = linear_probe.score(train_cell_states, train_depths)
    linear_probe_val_acc = linear_probe.score(val_cell_states, val_depths)
    linear_probe_val_full_acc = full_seq_accuracy(val_cell_states_grouped, val_depths_grouped, linear_probe)
    logger.info("Training Accuracy: {}".format(linear_probe_train_acc))
    logger.info("Validation Accuracy: {}".format(linear_probe_val_acc))
    logger.info("Validation Full Sequence Accuracy: {}".format(linear_probe_val_full_acc))

    logger.info("Training a Linear Probe on Control Task")
    linear_probe_control = LogisticRegression(multi_class='multinomial', random_state = 42, solver = 'lbfgs')
    linear_probe_control.fit(train_cell_states, train_random_labels)
    linear_probe_control_train_acc = linear_probe_control.score(train_cell_states, train_depths)
    linear_probe_control_val_acc = linear_probe_control.score(val_cell_states, val_depths)
    linear_probe_control_val_full_acc = full_seq_accuracy(val_cell_states_grouped, val_depths_grouped, linear_probe_control)
    logger.info("Training Accuracy: {}".format(linear_probe_control_train_acc))
    logger.info("Validation Accuracy: {}".format(linear_probe_control_val_acc))
    logger.info("Validation Full Sequence Accuracy: {}".format(linear_probe_control_val_full_acc))

    logger.info("Training an MLP Probe on Legit Data")
    mlp_probe = MLPClassifier(hidden_layer_sizes=(args.mlp_hid,), max_iter = 500)
    mlp_probe.fit(train_cell_states, train_depths)
    mlp_probe_train_acc = mlp_probe.score(train_cell_states, train_depths)
    mlp_probe_val_acc = mlp_probe.score(val_cell_states, val_depths)
    mlp_probe_val_full_acc = full_seq_accuracy(val_cell_states_grouped, val_depths_grouped, mlp_probe)
    logger.info("Training Accuracy: {}".format(mlp_probe_train_acc))
    logger.info("Validation Accuracy: {}".format(mlp_probe_val_acc))
    logger.info("Validation Full Sequence Accuracy: {}".format(mlp_probe_val_full_acc))

    logger.info("Training an MLP Probe on Control Task")
    mlp_probe_control = MLPClassifier(hidden_layer_sizes=(args.mlp_hid,), max_iter = 1000)
    mlp_probe_control.fit(train_cell_states, train_random_labels)
    mlp_probe_control_train_acc = mlp_probe_control.score(train_cell_states, train_depths)
    mlp_probe_control_val_acc = mlp_probe_control.score(val_cell_states, val_depths)
    mlp_probe_control_val_full_acc = full_seq_accuracy(val_cell_states_grouped, val_depths_grouped, mlp_probe_control)
    logger.info("Training Accuracy: {}".format(mlp_probe_control_train_acc))
    logger.info("Validation Accuracy: {}".format(mlp_probe_control_val_acc))
    logger.info("Validation Full Sequence Accuracy: {}".format(mlp_probe_control_val_full_acc))

    header = ['run_name', 'dataset', 'num_par', 'hidden_size', 'depth','rep_type', 'probe_layer', 'probe_type', 'probe_task', 'train_acc', 'val_acc', 'val_full_acc']
    row1 = [run_name, args.dataset, args.num_par, config.hidden_size, config.depth,rep_type, args.layer, 'Linear', 'Real', linear_probe_train_acc, linear_probe_val_acc, linear_probe_val_full_acc]
    row2 = [run_name, args.dataset, args.num_par, config.hidden_size, config.depth,rep_type, args.layer, 'Linear', 'Control', linear_probe_control_train_acc, linear_probe_control_val_acc, linear_probe_control_val_full_acc]
    row3 = [run_name, args.dataset, args.num_par, config.hidden_size, config.depth,rep_type, args.layer, 'MLP', 'Real', mlp_probe_train_acc, mlp_probe_val_acc, mlp_probe_val_full_acc]
    row4 = [run_name, args.dataset, args.num_par, config.hidden_size, config.depth, rep_type,args.layer, 'MLP', 'Control', mlp_probe_control_train_acc, mlp_probe_control_val_acc, mlp_probe_control_val_full_acc]

    results_file = os.path.join('out', 'probe_{}.csv'.format(args.dataset))
    mode = 'a' if os.path.exists(results_file) else 'w'
    with open(results_file, mode) as csvfile:
        writer = csv.writer(csvfile)
        if mode == 'a':
            rows = [row1, row2, row3, row4]
        else:
            rows = [header, row1, row2, row3, row4]
        for row in rows:
            writer.writerow(row)

    logger.info("Results stored at {}".format(results_file))
    




if __name__ == "__main__":
    main()