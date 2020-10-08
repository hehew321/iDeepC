import os, sys
import time
import argparse
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
import keras.backend.tensorflow_backend as KTF
from utils import generator_data, relu6, ABS, hard_swish


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


def train(train_x, train_y, use_transfer=True, save_model=True, dataset='RBP37', ):
    if dataset not in ['RBP24', 'RBP37']:
        print('Error entering dataset name, please enter RBP24 or RBP37')
    if dataset == 'RBP24':
        if use_transfer:
            model = tf.keras.models.load_model('RBP24_C22ORF28.h5',
                                               custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})
        else:
            model = my_model(shape=(train_x.shape[1], train_x.shape[2], 1))
    else:
        if use_transfer:
            model = tf.keras.models.load_model('RBP37_C22ORF28.h5',
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
                            callbacks=[earlystopper], steps_per_epoch=1*int(len(trainY) / batch_size), validation_steps=1)
    if save_model:
        model.save(protein_name + '.h5')
    return model


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
    train_x, train_y = load_data(protein_name=protein_name, pattern='train')
    model = train(train_x, train_y, use_transfer=True, save_model=True, dataset=dataset)

    print('use time:{}s '.format(time.time() - start_time))

