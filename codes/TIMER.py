# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
from keras import backend as K
import itertools
from OneHot_data import load_data
import argparse
import os,sys,re
import pandas as pd
from keras.models import load_model

def ABS(inputs):
    out1, out2 = inputs
    return abs(out1 - out2)

def relu6(x):
    return K.relu(x, max_value=6.0)

def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def binary(sequences):
    AA = 'ACGT'
    binary_feature = []
    for seq in sequences:
        binary = []
        for aa in seq:
            for aa1 in AA:
                tag = 1 if aa == aa1 else 0
                binary.append(tag)
        binary_feature.append(binary)
    return binary_feature

def read_fasta(inputfile, seq_type=None):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)
    data = {}
    for line in record:
        if line.startswith('>'):
            name = line.replace('>', '').split('\n')[0]
            data[name] = ''
        else:
            data[name] += line.replace('\n', '')
    if seq_type != None:
        return data
    else:
        sub_data = {}
        for key in data.keys():
            seq = data[key]
            if len(seq) <= 81:
                sub_data[key] = seq
            else:
                for j in range(0, len(seq) - 81 + 1):
                    sub_seq = seq[j:j + 81]
                    sub_data[key + '_' + str(j)] = sub_seq
        return sub_data


def extract_features(data):
    sequences = data
    feature_vector = np.vstack(binary(sequences))
    return feature_vector


def test(model, trainX, test_x, S):
    print('{}begin testing{}'.format('*' * 10, '*' * 10))
    scores = []
    pos_support = trainX
    if S == 2:#'C.jejuni' or 'C.pneumoniae' or 'H.pylori' or 'S.oneidensis' or 'General':
        pos_x = pos_support.reshape(pos_support.shape[0], pos_support.shape[1], pos_support.shape[2])
        for i, test_X in enumerate(test_x):
            repeat_test = test_X.reshape(1, test_X.shape[0], test_X.shape[1]).repeat(len(pos_x), axis=0)
            preds = model.predict([repeat_test, pos_x])
            scores.append(np.mean(preds))
        return scores
    else: #Species_name == 'B.amyloliquefaciens' or 'L.phytofermentans' or 'E.coli' or 'L.interrogans' or 'M. smegmatis' \
            #or 'R.capsulatus' or 'S.coelicolor' or 'S.pyogenes' or 'S.Typhimurium':
        pos_x = pos_support.reshape(pos_support.shape[0], pos_support.shape[1], pos_support.shape[2], 1)
        for i, test_X in enumerate(test_x):
            repeat_test = test_X.reshape(1, test_X.shape[0], test_X.shape[1], 1).repeat(len(pos_x), axis=0)
            preds = model.predict([repeat_test, pos_x])
            scores.append(np.mean(preds))
        return scores


def calculate_performace(y_pred):
    labels = []
    for i in range(len(y_pred)):
        if y_pred[i] >= 0.5:
            labels.append(1)
        if y_pred[i] < 0.5:
            labels.append(0)
    return labels


def main():
    parser = argparse.ArgumentParser(
        description='TIMER: a deep-learning framework for general and species-specific bacterial promoter prediction')
    parser.add_argument('--input', dest='inputfile', type=str, required=True,
                        help='query sequences to be predicted in fasta format.')
    parser.add_argument('--output', dest='outputfile', type=str, required=False, help='save the prediction results.')
    parser.add_argument('--seq_type', dest='seq_type', type=str, required=True, choices=['full_length', 'fixed_length'])
    parser.add_argument('--species', dest='speciesfile', type=str, required=False,
                        help='--species indicates the specific species, currently we accept \'B.amyloliquefaciens\' or\'C. jejuni\' or \'L. phytofermentans\' or \'C. pneumoniae\' \n \or '
                             'E. coli\' or\'H. pylori\' or\'L. interrogans\' or\'M. smegmatis\' or\'R. capsulatus\' or\'S. coelicolor\' or\'S. oneidensis\' or\'S. pyogenes\' or\'S. Typhimurium\' or\'General.\n \
                        ', default=None)
    args = parser.parse_args()

    inputfile = args.inputfile
    outputfile = args.outputfile
    speciesfile = args.speciesfile

    seqtype = None
    if args.seq_type == 'fixed_length':
        seqtype = 'fl'
    data = read_fasta(inputfile, seqtype)

    vector = extract_features(data.values())
    df = pd.DataFrame(vector)
    feature_names = []
    for i in range(0, len(df.columns)):
        feature_names.append(df.columns[i])
    ppp = df[feature_names]
    ppp = ppp.values.reshape(len(ppp), 81, 4)
    outputfile_original = outputfile
    if outputfile_original == None:
        outputfile = 'output'
    if speciesfile == 'B.amyloliquefaciens':
        Species_name = 'B.amyloliquefaciens'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/B.amyloliquefaciens.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'C.jejuni':
        Species_name = 'C.jejuni'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/C.jejuni.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 2)
    elif speciesfile == 'C.pneumoniae':
        Species_name = 'C.pneumoniae'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/C.pneumoniae.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 2)
    elif speciesfile == 'E.coli':
        Species_name = 'E.coli'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/E.coli.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'H.pylori':
        Species_name = 'H.pylori'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/H.pylori.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 2)
    elif speciesfile == 'L.interrogans':
        Species_name = 'L.interrogans'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/L.interrogans.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'L.phytofermentans':
        Species_name = 'L.phytofermentans'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/L.phytofermentans.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'M.smegmatis':
        Species_name = 'M.smegmatis'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/M.smegmatis.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'R.capsulatus':
        Species_name = 'R.capsulatus'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/R.capsulatus.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'S.coelicolor':
        Species_name = 'S.coelicolor'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/S.coelicolor.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'S.oneidensis':
        Species_name = 'S.oneidensis'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/S.oneidensis.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'S.pyogenes':
        Species_name = 'S.pyogenes'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/S.pyogenes.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'S.Typhimurium':
        Species_name = 'S.Typhimurium'
        trainX = load_data(Species_name=Species_name, pattern='train_all')
        model = load_model("../models/S.Typhimurium.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 1)
    elif speciesfile == 'General':
        trainX = load_data(Species_name='general', pattern='train_all')
        model = load_model("../models/General.h5",custom_objects = {'relu6': relu6, 'hard_swish': hard_swish})
        predictions = test(model, trainX, ppp, 2)

    probability = ['%.5f' % float(i) for i in predictions]
    name = list(data.keys())
    seq = list(data.values())
    decisions = ['%.5f' % float(i) for i in predictions]
    with open(outputfile, 'w') as f:
        for i in range(len(data)):
            if float(probability[i]) > 0.5:
                f.write(probability[i] + '*' + '\t')
                f.write(decisions[i] + '\t')
                f.write(name[i] + '\t')
                f.write(seq[i] + '\n')
            else:
                f.write(probability[i] + '\t')
                f.write(name[i] + '\t')
                f.write(seq[i] + '\n')
    print('output are saved in ' + outputfile + ', and those identified as promoters are marked with *')

if __name__ == "__main__":
    main()


