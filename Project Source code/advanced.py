import itertools
from keras.layers import Activation, Reshape, Lambda, concatenate, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras.callbacks import Callback, TensorBoard
from keras.engine.topology import Layer
from keras import backend as K

''' Callbacks '''
class HistoryCheckpoint(Callback):
  
    def __init__(self, filename):
        super(Callback, self).__init__()
        self.filename = filename

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        for k, v in logs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

        with open(self.filename, "w") as f:
            f.write(str(self.history))

class TensorBoardBatch(TensorBoard):
    def __init__(self, log_dir='./logs',
                 histogram_freq=0,
                 batch_size=32,
                 write_graph=True,
                 write_grads=False,
                 write_images=False,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None):
        super(TensorBoardBatch, self).__init__(log_dir,
                                               histogram_freq=histogram_freq,
                                               batch_size=batch_size,
                                               write_graph=write_graph,
                                               write_grads=write_grads,
                                               write_images=write_images,
                                               embeddings_freq=embeddings_freq,
                                               embeddings_layer_names=embeddings_layer_names,
                                               embeddings_metadata=embeddings_metadata)

        # conditionally import tensorflow iff TensorBoardBatch is created
        self.tf = __import__('tensorflow')
        self.global_step = 1

class SubPixelUpscaling(Layer):

    def __init__(self, r, channels, **kwargs):
        super(SubPixelUpscaling, self).__init__(**kwargs)

        self.r = r
        self.channels = channels

    def build(self, input_shape):
        pass

 
    def get_output_shape_for(self, input_shape):
        
        b, r, c, k = input_shape
        return (b, r * self.r, c * self.r, self.channels)


''' Non Local Blocks '''

def non_local_block(ip, computation_compression=2, mode='embedded'):
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    dim1, dim2, dim3 = None, None, None

    if len(ip_shape) == 3:  # time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # Video / Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        f = dot([theta, phi], axes=2)

        # scale the values to make it size invariant
        if batchsize is not None:
            f = Lambda(lambda z: 1./ batchsize * z)(f)
        else:
            f = Lambda(lambda z: 1. / 128 * z)(f)


    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplemented('Concatenation mode has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, channels // 2)
        theta = Reshape((-1, channels // 2))(theta)

        # phi path
        phi = _convND(ip, rank, channels // 2)
        phi = Reshape((-1, channels // 2))(phi)

        if computation_compression > 1:
            # shielded computation
            phi = MaxPool1D(computation_compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, channels // 2)
    g = Reshape((-1, channels // 2))(g)

    if computation_compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(computation_compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, channels // 2))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, channels // 2))(y)
        else:
            y = Reshape((channels // 2, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    residual = add([ip, y])

    return residual


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False)(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False)(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False)(ip)
    return x
