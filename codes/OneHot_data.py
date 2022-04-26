import numpy as np

def load_data(Species_name, pattern):
    seq = []
    fp1 = open('train-data/' + Species_name + '/' + pattern + '_p.txt')
    for line in fp1:
        if line[0] != '>':
            seq.append(replace_seq(line.strip()))

    seq_data = to_matrix(seq)
    return seq_data

def replace_seq(seq):
    seq = seq.replace('T', 'U')
    seq = seq.replace('W', 'U')
    seq = seq.replace('N', 'U')
    seq = seq.replace('u', 'U')
    seq = seq.replace('t', 'U')
    seq = seq.replace('a', 'A')
    seq = seq.replace('g', 'G')
    seq = seq.replace('c', 'C')
    return seq


def to_matrix(seq):
    row_number = 81
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
