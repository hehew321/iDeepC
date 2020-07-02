import os, sys
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as KTF
from utils import get_feature, get_seq, get_logo


def cnn_block(input):
    out = tf.keras.layers.Conv2D(64, kernel_size=(7, 4), strides=1,
                                 activation='relu', padding="valid")(input)

    out = tf.keras.layers.Conv2D(32, kernel_size=(7, 1), strides=1,
                                 activation='relu', padding="valid")(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(3, 1), strides=1)(out)
    out = tf.keras.layers.Dropout(0.25)(out)


    channel = int(out.shape[-1])  
    x = tf.keras.layers.GlobalAveragePooling2D()(out)
    x = tf.keras.layers.Dense(int(channel / 4), activation=relu6)(x)
    x = tf.keras.layers.Dense(channel, activation=hard_swish)(x)
    x = tf.keras.layers.Reshape((1, 1, channel))(x)
    x = tf.keras.layers.Multiply()([out, x])

    out = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(64, activation='sigmoid')(out)
    return out


def my_model(shape):
    input_tensor = tf.keras.Input(shape=shape)  
    cnn_model = tf.keras.Model(input_tensor, cnn_block(input_tensor))
    input1 = tf.keras.Input(shape=shape)
    input2 = tf.keras.Input(shape=shape)
    out1 = cnn_model(input1)
    out2 = cnn_model(input2)

    out = tf.keras.layers.Lambda(ABS)([out1, out2])
    out = tf.keras.layers.Dense(1, activation='sigmoid')(out)
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
    return model


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
            if count_1 < 33+int(len(data_y)/4000):
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
    return


def train(train_x, train_y, use_transfer=True, save_model=None, dataset='CRIP', ):
    if dataset not in ['CRIP', 'GraphProt_CLIP']:
        print('Error entering dataset name, please enter CRIP or GraphProt_CLIP')
    if dataset == 'CRIP':
        if use_transfer:
            model = tf.keras.models.load_model('CRIP_C22ORF28.h5',
                                               custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})
        else:
            model = my_model(shape=(train_x.shape[1], train_x.shape[2], 1))
    else:
        if use_transfer:
            model = tf.keras.models.load_model('GraphProt_CLIP_C22ORF28_Baltz2012.h5',
                                               custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})

        else:
            model = my_model(shape=(train_x.shape[1], train_x.shape[2], 1))
    batch_size, epochs = 128, 20
    kf = KFold(n_splits=5, random_state=222)

    for train_index, eval_index in kf.split(train_x, train_y):
        trainX = train_x[train_index]
        trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
        evalX = train_x[eval_index]
        evalX = evalX.reshape(evalX.shape[0], evalX.shape[1], evalX.shape[2], 1)
        trainY = train_y[train_index]
        evalY = train_y[eval_index]

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-4))
        model.fit_generator(generator_data(x=trainX, y=trainY, b_size=batch_size), verbose=2,
                            epochs=epochs, validation_data=generator_data(x=evalX, y=evalY, b_size=batch_size),
                            callbacks=[earlystopper], steps_per_epoch=int(len(trainY) / batch_size), validation_steps=1)
    if save_model:
        model.save('./' + protein_name+'.h5')
    return model


def test(model, test_x, test_y, draw_motifs=None, protein_name='ALKBH5'):
    print('{}开始测试{}'.format('*'*10, '*'*10))
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
    parser.add_argument('--dataset', type=str, default='CRIP', help='Select CRIP or GraphProt_CLIP')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print('{} start training {}'.format('*'*10, '*'*10))

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
    if dataset not in ['CRIP', 'GraphProt_CLIP']:
        print('Error: The name of the data set was entered incorrectly, please enter CRIP or GraphProt_CLIP')
        sys.exit()
    if dataset == 'CRIP':
        from CRIP_data import load_data
    else:
        from GraphProt_CLIP_data import load_data

    print('protein_name: ', protein_name)
    train_x, train_y = load_data(protein_name=protein_name, pattern='train')
    test_x, test_y = load_data(protein_name=protein_name, pattern='ls')
    model = train(train_x, train_y, use_transfer=True, save_model=None, dataset=dataset)
    test(model, test_x, test_y, draw_motifs=False, protein_name=protein_name)

    print('use time:{}s '.format(time.time()-start_time))

