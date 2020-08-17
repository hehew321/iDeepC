import numpy as np


def load_data(protein_name, pattern='train'):
    seq = []
    seq_label = []

    fp1 = open('./datasets/CRIP_split/' + protein_name + '/' + protein_name + '.' + pattern + '.' + 'positive')
    fp2 = open('./datasets/CRIP_split/' + protein_name + '/' + protein_name + '.' + pattern + '.' + 'negative')
    for line in fp1:
        if line[0] != '>':
            seq.append(replace_seq(line.strip()))
            seq_label.append(1)

    for line in fp2:
        if line[0] != '>':
            seq.append(replace_seq(line.strip()))
            seq_label.append(0)

    indexes = np.random.choice(len(seq), len(seq), replace=False)
    seq = np.array(seq)[indexes]
    seq_label = np.array(seq_label)[indexes]

    seq_data = to_matrix(seq)

    return seq_data, seq_label


def replace_seq(seq):
    seq = seq.replace('T', 'U')
    seq = seq.replace('u', 'U')
    seq = seq.replace('t', 'U')
    seq = seq.replace('a', 'A')
    seq = seq.replace('g', 'G')
    seq = seq.replace('c', 'C')
    return seq


def to_matrix(seq):
    row_number = 101
    seq_data = []

    for i in range(len(seq)):
        mat = np.array([0.] * 4 * row_number).reshape(row_number, 4)
        for j in range(len(seq[i])):

            if seq[i][j] == 'A':
                mat[j][0] = 1.0
            elif seq[i][j] == 'C':
                mat[j][1] = 1.0
            elif seq[i][j] == 'G':
                mat[j][2] = 1.0
            elif seq[i][j] == 'U':
                mat[j][3] = 1.0
        seq_data.append(mat)
    return np.array(seq_data)


if __name__ == '__main__':
    train_xx, train_yy = load_data("ALKBH5")
    print(train_xx[0])
    print(len(train_yy))

