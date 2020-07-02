import random
import numpy as np


def load_data(protein_name='ALKBH5_Baltz2012', pattern='train'):
    seq = []
    data_y = []
    for sample in ['positives', 'negatives']:
        f = open('./GraphProt_CLIP_sequences/' + protein_name + '.' + pattern + '.' + sample + '.fa')
        lines_list = []
        for line in f:
            lines_list.append(line)
        for i in range(len(lines_list)):
            if lines_list[i][0] == '>':
                tem_seq = lines_list[i + 1].split()
                seq.append(tem_seq)
                if sample == 'positives':
                    data_y.append(1)
                else:
                    data_y.append(0)
    assert len(data_y) == len(seq)
    data_x = to_matrix(seq)
    random.seed(222)
    random.shuffle(data_x)
    random.seed(222)
    random.shuffle(data_y)

    return np.array(data_x), np.array(data_y)


def pos_sample(protein_name='ALKBH5_Baltz2012'):
    f = open('./GraphProt_CLIP_sequences/' + protein_name + '.train.positives.fa')
    lines_list = []
    seq = []
    for line in f:
        lines_list.append(line)
    for i in range(len(lines_list)):
        if lines_list[i][0] == '>':
            tem_seq = lines_list[i + 1].split()
            seq.append(tem_seq)
    seq_mat = to_matrix(seq)
    return np.array(seq_mat)


def to_matrix(seq):
    seq_mat = []
    row_number = 380
    for i in range(len(seq)):
        mat = np.array([0.] * 4 * row_number).reshape(row_number, 4)
        for j in range(len(seq[i][0])):
            if seq[i][0][j] == 'A' or seq[i][0][j] == 'a':
                mat[j][0] = 1.0
            elif seq[i][0][j] == 'T' or seq[i][0][j] == 't' or \
                    seq[i][0][j] == 'u' or seq[i][0][j] == 'U':
                mat[j][1] = 1.0
            elif seq[i][0][j] == 'C' or seq[i][0][j] == 'c':
                mat[j][2] = 1.0
            elif seq[i][0][j] == 'G' or seq[i][0][j] == 'g':
                mat[j][3] = 1.0
        seq_mat.append(mat)
    return seq_mat





