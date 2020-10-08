import os, sys
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import keras.backend.tensorflow_backend as KTF
from utils import get_feature, get_seq, get_logo, relu6, ABS, get_pos, hard_swish


def get_motifs(model, data_x, data_y, protein_name):
    model.trainable = False
    pos_x = get_pos(data_x, data_y)
    data_x = pos_x.reshape(pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], 1)
    features = get_feature(model=model, data=data_x)
    features = features.reshape(features.shape[0], features.shape[1], features.shape[3])
    if os.path.exists('./' + protein_name + '.fa'):
        os.remove('./' + protein_name + '.fa')
    fp = open('./' + protein_name + '.fa', 'w')

    count = 0
    for i in range(features.shape[0]):
        seq = get_seq(data_x[i])
        for j in range(features.shape[1]):
            count_1 = 0
            for k in range(features.shape[2]):
                if features[i][j][k] > 0:
                    count_1 += 1
            if count_1 < 33 + int(len(data_y) / 4000):
                continue
            else:
                for k in range(features.shape[2]):
                    if features[i][j][k] > 0.4:
                        fp.write('>' + 'seq_' + str(i) + '_' + 'filter' + str(j) + '_' + str(k) + '\n')
                        fp.write(seq[j:j + 7] + '\n')
                        count += 1
    fp.close()
    print('count：', count)
    print('{}start get {} logo{}'.format('*' * 10, protein_name, '*' * 10))
    get_logo(protein_name)
    print('{}draw {} logo done{}'.format('*' * 10, protein_name, '*' * 10))


def test(model, test_x, test_y, draw_motifs=None, protein_name='ALKBH5'):
    print('{}begin testing{}'.format('*' * 10, '*' * 10))
    scores = []
    pos_support = get_pos(train_x, train_y)

    pos_x = pos_support.reshape(pos_support.shape[0], pos_support.shape[1], pos_support.shape[2], 1)

    for i, test_X in enumerate(test_x):
        repeat_test = test_X.reshape(1, test_X.shape[0], test_X.shape[1], 1).repeat(len(pos_x), axis=0)
        preds = model.predict([repeat_test, pos_x])
        scores.append(np.mean(preds))
    AUCs = roc_auc_score(test_y, scores)
    print("Mean AUC: {}, protein name: {}".format(np.mean(AUCs), protein_name))
    if draw_motifs:
        get_motifs(model, test_x, test_y, protein_name)


def parse_arguments(parser):
    parser.add_argument('--protein_name', type=str, default='ALKBH5', help='Enter the name of the protein')
    parser.add_argument('--dataset', type=str, default='RBP37', help='Select RBP24 or RBP37')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('{} start training {}'.format('*' * 10, '*' * 10))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    start_time = time.time()

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    print(args)
    protein_name = args.protein_name
    dataset = args.dataset
    if dataset not in ['RBP24', 'RBP37']:
        print('Error: The name of the data set was entered incorrectly, please enter RBP24 or RBP37')
        sys.exit()
    if dataset == 'RBP24':
        from RBP24_data import load_data
    else:
        from RBP37_data import load_data

    print('protein_name: ', protein_name)

    test_x, test_y = load_data(protein_name=protein_name, pattern='ls')
    train_x, train_y = load_data(protein_name=protein_name, pattern='train')

    model = tf.keras.models.load_model(protein_name+'.h5',
                                       custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})
    test(model, test_x, test_y, draw_motifs=False, protein_name=protein_name)

    print('use time:{}s '.format(time.time() - start_time))

