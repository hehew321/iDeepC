from __future__ import division, print_function
import numpy as np
from time import sleep
import sys
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from data import load_data
from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements the paper "Axiomatic attribution
for deep neuron networks".
'''

def ABS(inputs):
    out1, out2 = inputs
    return tf.abs(out1-out2)


def relu6(x):
    return tf.keras.backend.relu(x, max_value=6.0)


def swish(x):
    return x * tf.keras.backend.sigmoid(x)


def hard_swish(x):
    return x * tf.keras.backend.relu(x+3.0, max_value=6.0)/6.0


class integrated_gradients:
    def __init__(self, model, outchannels=[], verbose=1):

        self.backend = K.backend()

        if isinstance(model, Sequential):
            self.model = model.model
        else:
            self.model = model

        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        self.input_tensors.append(K.learning_phase())

        self.outchannels = outchannels
        if len(self.outchannels) == 0:
            if verbose: print("Evaluated output channel (0-based index): All")
            if K.backend() == "tensorflow":
                self.outchannels = range(self.model.output.shape[1]._value)
            elif K.backend() == "theano":
                self.outchannels = range(self.model.output._keras_shape[1])
        else:
            if verbose:
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))

        self.get_gradients = {}
        if verbose: print("Building gradient functions")

        print('self.outchannels: ', self.outchannels)
        for c in self.outchannels:
            # Get tensor that calculates gradient
            if K.backend() == "tensorflow":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
            if K.backend() == "theano":
                gradients = self.model.optimizer.get_gradients(self.model.output[:, c].sum(), self.model.input)

            self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: " + str(int((c + 1) * 1.0 / len(self.outchannels) * 1000) * 1.0 / 10) + "%")
                sys.stdout.flush()
        # Done
        if verbose: print("\nDone.")


    def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):

        samples = []
        numsteps = []
        step_sizes = []

        if isinstance(sample, list):
            if reference != False:
                assert len(sample) == len(reference)
            for i in range(len(sample)):
                if reference == False:
                    _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
                else:
                    _output = integrated_gradients.linearly_interpolate(sample[i], reference[i], num_steps)
                samples.append(_output[0])
                numsteps.append(_output[1])
                step_sizes.append(_output[2])

        elif isinstance(sample, np.ndarray):
            _output = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
            samples.append(_output[0])
            numsteps.append(_output[1])
            step_sizes.append(_output[2])

        if verbose: print("Explaning the " + str(self.outchannels[outc]) + "th output.")

        # For tensorflow backend
        __input = []
        for s in samples:
            __input.append(s)
        __input.append(0)
        if K.backend() == "tensorflow":

            gradients = self.get_gradients[0](__input)
        elif K.backend() == "theano":
            gradients = self.get_gradients[outc](__input)
            if len(self.model.inputs) == 1:
                gradients = [gradients]

        explanation = []
        for i in range(len(gradients)):
            _temp = np.sum(gradients[i], axis=0)
            explanation.append(np.multiply(_temp, step_sizes[i]))

        # Format the return values according to the input sample.
        if isinstance(sample, list):
            return explanation
        elif isinstance(sample, np.ndarray):
            return explanation[0]
        return -1

    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        if reference is False: reference = np.zeros(sample.shape);

        assert sample.shape == reference.shape
        ret = np.zeros(tuple([num_steps] + [i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference + (sample - reference) * (s * 1.0 / num_steps)

        return ret, num_steps, (sample - reference) * (1.0 / num_steps)


def plot_integratedGradients(ig, xx1, xx2, yy, pred, index):
    _font1 = {'family': 'Times New Roman',
              'weight': 'normal',
              'size': 4,
              }
    ex = ig.explain([xx1[index, :, :, :], xx2[index, :, :, :]], outc=pred[index])
    th = max(np.abs(np.min([np.min(ex[0]), np.min(ex[1])])), np.abs(np.max([np.max(ex[0]), np.max(ex[1])])))
    seq1 = get_seq(xx1[index])
    seq2 = get_seq(xx2[index])

    tem_seq = ''
    for ii in seq1:
        tem_seq += ii
    print('ex[0][:, :, 0]: ', np.array(ex[0][:, :, 0]).tolist())
    print('ex[1][:, :, 0]: ', np.array(ex[1][:, :, 0]).tolist())
    # print('seq1: ', tem_seq)
    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title(str(yy[index]))
    im1 = plt.imshow(np.transpose(ex[0][:, :, 0]), cmap="seismic", vmin=-1 * th, vmax=th)
    plt.colorbar(im1, cax=None, ax=None, shrink=1)
    plt.yticks([], [])
    plt.xticks(range(101), seq1)
    plt.tick_params(labelsize=7)
    # plt.show()

    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 2)
    im2 = plt.imshow(np.transpose(ex[1][:, :, 0]), cmap="seismic", vmin=-1 * th, vmax=th)
    plt.colorbar(im2, cax=None, ax=None, shrink=1)
    plt.yticks([], [])
    plt.xticks(range(101), seq2)
    plt.tick_params(labelsize=7)
    plt.show()


def get_seq(x):
    seq = []
    x = x.reshape(x.shape[0], x.shape[1])
    for i in range(len(x)):
        if x[i][0] == 1.0:
            seq.append('A')
        elif x[i][1] == 1.0:
            seq.append('C')
        elif x[i][2] == 1.0:
            seq.append('G')
        elif x[i][3] == 1.0:
            seq.append('U')
    return seq


def run_IntegratedGradients(model, X, Y):
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    index1 = np.random.choice(X.shape[0], 2*X.shape[0], replace=True)
    index2 = np.random.choice(X.shape[0], 2*X.shape[0], replace=True)
    xx1 = X[index1]
    xx2 = X[index2]
    Y1 = Y[index1]
    Y2 = Y[index2]
    print('xx2.shape: ', xx2.shape)
    yy = []
    for ii in range(len(index1)):
        if Y[index1[ii]] == 1 and Y[index2[ii]] == 1 and index1[ii] != index2[ii]:
            yy.append(1)
        else:
            yy.append(0)
    pred = model.predict([xx1, xx2])
    ig = integrated_gradients(model)
    i = 0
    for _index in range(1, 20):
        index = _index+10
        if Y1[index] == 1 and Y2[index] == 1:
            i += 1
            plot_integratedGradients(ig, xx1, xx2, yy, pred, index)


if __name__ == '__main__':
    protein_name = ''
    train_x, train_y = load_data(protein_name, pattern='train')
    test_x, test_y = load_data(protein_name, pattern='ls')
    model = tf.keras.models.load_model('../iDeepC_new_result/RBP37_saved_models/' + protein_name + '.h5',
                                       custom_objects={'relu6': relu6, 'ABS': ABS, 'hard_swish': hard_swish})
    run_IntegratedGradients(model=model, X=train_x, Y=train_y)