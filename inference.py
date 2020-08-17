import numpy as np
import tensorflow as tf
from data import test_load
import matplotlib.pyplot as plt


def ABS(inputs):
    out1, out2 = inputs
    return tf.abs(out1-out2)


def relu6(x):
    return tf.keras.backend.relu(x, max_value=6.0)


def swish(x):
    return x * tf.keras.backend.sigmoid(x)


def hard_swish(x):
    return x * tf.keras.backend.relu(x+3.0, max_value=6.0)/6.0


def get_pos_x(x, y):
    pos_indexes = []
    for i in range(len(y)):
        if y[i] == 1:
            pos_indexes.append(i)

    return x[np.array(pos_indexes)]


if __name__ == '__main__':
    protein_name = 'AUF1'
    model = tf.keras.models.load_model('./saved_models/'+protein_name+'.h5',
                                       custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})
    support_x, support_y = test_load(protein_name, pattern='train')
    test_x, test_y = test_load(protein_name, pattern='ls')
    print(len(support_x), len(support_y))
    pos_x = support_x.reshape(support_x.shape[0], support_x.shape[1], support_x.shape[2], 1)

    print('{}开始测试{}'.format('*' * 10, '*' * 10))
    scores = []
    for i, test_X in enumerate(test_x):
        tem = []
        repeat_test = test_X.reshape(1, test_X.shape[0], test_X.shape[1], 1).repeat(len(pos_x), axis=0)
        preds = model.predict([repeat_test, pos_x])

        scores.append(np.mean(preds))

    print("protein name: {}, scores: {}".format(protein_name, scores))

    plt.bar([i for i in range(len(scores))], scores)
    # plt.title('hsa_circ_0000006')
    plt.show()
