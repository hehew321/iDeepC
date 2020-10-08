import tensorflow as tf
from weblogo import *



def ABS(inputs):
    out1, out2 = inputs
    return tf.abs(out1-out2)


def relu6(x):
    return tf.keras.backend.relu(x, max_value=6.0)


def swish(x):
    return x * tf.keras.backend.sigmoid(x)


def hard_swish(x):
    return x * tf.keras.backend.relu(x+3.0, max_value=6.0)/6.0


def get_pos(x, y):
    pos_indexes = []
    for i in range(len(y)):
        if y[i] == 1:
            pos_indexes.append(i)

    return x[np.array(pos_indexes)]


def get_feature(model, data):
    inputs = [tf.keras.backend.learning_phase()] + [model.layers[2].inputs]
    _convout1_f = tf.keras.backend.function(inputs, model.layers[2].layers[1].output)

    return _convout1_f([[0]+[data]])


def get_seq(x):
    seq = ''
    x = x.reshape(x.shape[0], x.shape[1])
    for i in range(len(x)):
        if x[i][0] == 1.0:
            seq += 'A'
        elif x[i][1] == 1.0:
            seq += 'C'
        elif x[i][2] == 1.0:
            seq += 'G'
        elif x[i][3] == 1.0:
            seq += 'U'
    return seq


def get_logo(protein_name):
    f = open(protein_name+'.fa', 'r')

    # 获取序列数据
    seqs = read_seq_data(f)
    data = LogoData.from_seqs(seqs)
    options = LogoOptions()
    options.logo_title = protein_name
    options.fineprint = ''
    color_scheme = ColorScheme(
        [
            SymbolColor("G", "#FBB116"),
            SymbolColor("U", "#CB2026"),
            SymbolColor("C", "#34459C"),
            SymbolColor("A", "#0C8040")
        ])
    options.color_scheme = color_scheme
    format = LogoFormat(data, options)
    eps = eps_formatter(data, format)
    with open(protein_name+'.eps', 'wb') as fp:
        fp.write(eps)
        fp.close()
    f.close()


def generator_data(x, y, b_size):
    while True:
        index1 = np.random.choice(x.shape[0], b_size, replace=False)
        index2 = np.random.choice(x.shape[0], b_size, replace=True)
        xx1 = x[index1]
        xx2 = x[index2]
        yy = []
        for ii in range(b_size):
            if y[index1[ii]] == 1 and y[index2[ii]] == 1 and index1[ii] != index2[ii]:
                yy.append(1)
            else:
                yy.append(0)

        yield [xx1, xx2], np.array(yy)
