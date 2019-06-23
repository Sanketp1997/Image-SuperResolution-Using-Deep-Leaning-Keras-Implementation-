from __future__ import print_function, division
from keras.models import Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Convolution2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers
from advanced import HistoryCheckpoint, SubPixelUpscaling, non_local_block, TensorBoardBatch
import img_utils
import numpy as np
import os
import time
import warnings

try:
    import cv2
    _cv2_available = True
except:
    warnings.warn('Could not load opencv properly. This may affect the quality of output images.')
    _cv2_available = False

train_path = img_utils.output_path
validation_path = img_utils.validation_output_path
path_X = img_utils.output_path + "X/"
path_Y = img_utils.output_path + "y/"

def PSNRLoss(y_true, y_pred):
  
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

def psnr(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return -10. * np.log10(np.mean(np.square(y_pred - y_true)))

class BaseSuperResolutionModel(object):

    def __init__(self, model_name, scale_factor):
      
        self.model = None # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None

        self.type_scale_type = "norm" # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128) -> Model:
      
        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height * img_utils._image_scale_multiplier % 4 == 0, "Height of the image must be divisible by 4"
            assert width * img_utils._image_scale_multiplier % 4 == 0, "Width of the image must be divisible by 4"

        if K.image_dim_ordering() == "th":
            if width is not None and height is not None:
                shape = (channels, width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier)
            else:
                shape = (channels, None, None)
        else:
            if width is not None and height is not None:
                shape = (width * img_utils._image_scale_multiplier, height * img_utils._image_scale_multiplier, channels)
            else:
                shape = (None, None, channels)

        init = Input(shape=shape)

        return init

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Model History.txt") -> Model:
        
        samples_per_epoch = img_utils.image_count()
        val_count = img_utils.val_image_count()
        if self.model == None: self.create_model(batch_size=batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNRLoss', save_best_only=True,
                                                   mode='max', save_weights_only=True, verbose=2)]
        if save_history:
            callback_list.append(HistoryCheckpoint(history_fn))

            if K.backend() == 'tensorflow':
                log_dir = './%s_logs/' % self.model_name
                tensorboard = TensorBoardBatch(log_dir, batch_size=batch_size)
                callback_list.append(tensorboard)

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(img_utils.image_generator(train_path, scale_factor=self.scale_factor,
                                                           small_train_images=self.type_true_upscaling,
                                                           batch_size=batch_size),
                                 steps_per_epoch=samples_per_epoch // batch_size + 1,
                                 epochs=nb_epochs, callbacks=callback_list,
                                 validation_data=img_utils.image_generator(validation_path,
                                                                           scale_factor=self.scale_factor,
                                                                           small_train_images=self.type_true_upscaling,
                                                                           batch_size=batch_size),
                                 validation_steps=val_count // batch_size + 1)

        return self.model
  
    def upscale(self, img_path, save_intermediate=False, return_image=False, suffix="scaled",
                patch_size=8, mode="patch", verbose=True):
        
        import os
        from scipy.misc import imread, imresize, imsave

        # Destination path
        path = os.path.splitext(img_path)
        #print(path[0])
        filename = path[0] + "_" + suffix + "(%dx)" % (self.scale_factor) + path[1]
        
        # Read image
        scale_factor = int(self.scale_factor)
        true_img = imread(img_path, mode='RGB')
        init_dim_1, init_dim_2 = true_img.shape[0], true_img.shape[1]
        if verbose: print("Old Size : ", true_img.shape)
        if verbose: print("New Size : (%d, %d, 3)" % (init_dim_1 * scale_factor, init_dim_2 * scale_factor))

        img_dim_1, img_dim_2 = 0, 0

        if mode == "patch" and self.type_true_upscaling:
            # Overriding mode for True Upscaling models
            mode = 'fast'
            print("Patch mode does not work with True Upscaling models yet. Defaulting to mode='fast'")

        if mode == 'patch':
            # Create patches
            if self.type_requires_divisible_shape:
                if patch_size % 4 != 0:
                    print("Deep Denoise requires patch size which is multiple of 4.\nSetting patch_size = 8.")
                    patch_size = 8

            images = img_utils.make_patches(true_img, scale_factor, patch_size, verbose)

            nb_images = images.shape[0]
            img_dim_1, img_dim_2 = images.shape[1], images.shape[2]
            print("Number of patches = %d, Patch Shape = (%d, %d)" % (nb_images, img_dim_2, img_dim_1))
        else:
            # Use full image for super resolution
            img_dim_1, img_dim_2 = self.__match_autoencoder_size(img_dim_1, img_dim_2, init_dim_1, init_dim_2,
                                                                 scale_factor)

            images = imresize(true_img, (img_dim_1, img_dim_2))
            images = np.expand_dims(images, axis=0)
            print("Image is reshaped to : (%d, %d, %d)" % (images.shape[1], images.shape[2], images.shape[3]))

        # Save intermediate bilinear scaled image is needed for comparison.
        # intermediate_img = None
        # if save_intermediate:
        #     if verbose: print("Saving intermediate image.")
        #     fn = path[0] + "_intermediate_" + path[1]
        #     intermediate_img = imresize(true_img, (init_dim_1 * scale_factor, init_dim_2 * scale_factor))
        #     imsave(fn, intermediate_img)

        # Transpose and Process images
        if K.image_dim_ordering() == "th":
            img_conv = images.transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        else:
            img_conv = images.astype(np.float32) / 255.

        model = self.create_model(img_dim_2, img_dim_1, load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict(img_conv, batch_size=128, verbose=verbose)

        if verbose: print("De-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        # Output shape is (original_width * scale, original_height * scale, nb_channels)
        if mode == 'patch':
            out_shape = (init_dim_1 * scale_factor, init_dim_2 * scale_factor, 3)
            result = img_utils.combine_patches(result, out_shape, scale_factor)
        else:
            result = result[0, :, :, :] # Access the 3 Dimensional image vector

        result = np.clip(result, 0, 255).astype('uint8')

        if _cv2_available:
            # used to remove noisy edges
            result = cv2.pyrUp(result)
            result = cv2.medianBlur(result, 3)
            result = cv2.pyrDown(result)

        if verbose: print("\nCompleted De-processing image.")

        if return_image:
            # Return the image without saving. Useful for testing images.
            return result

        if verbose: print("Saving image.")
        imsave(filename, result)

    def __match_autoencoder_size(self, img_dim_1, img_dim_2, init_dim_1, init_dim_2, scale_factor):
        if self.type_requires_divisible_shape:
            if not self.type_true_upscaling:
                # AE model but not true upsampling
                if ((init_dim_2 * scale_factor) % 4 != 0) or ((init_dim_1 * scale_factor) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2 * scale_factor) // 4) * 4
                    img_dim_1 = ((init_dim_1 * scale_factor) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor
            else:
                # AE model and true upsampling
                if ((init_dim_2) % 4 != 0) or ((init_dim_1) % 4 != 0) or \
                        (init_dim_2 % 2 != 0) or (init_dim_1 % 2 != 0):

                    print("AE models requires image size which is multiple of 4.")
                    img_dim_2 = ((init_dim_2) // 4) * 4
                    img_dim_1 = ((init_dim_1) // 4) * 4

                else:
                    # No change required
                    img_dim_2, img_dim_1 = init_dim_2, init_dim_1
        else:
            # Not AE but true upsampling
            if self.type_true_upscaling:
                img_dim_2, img_dim_1 = init_dim_2, init_dim_1
            else:
                # Not AE and not true upsampling
                img_dim_2, img_dim_1 = init_dim_2 * scale_factor, init_dim_1 * scale_factor

        return img_dim_1, img_dim_2,


class DistilledResNetSR(BaseSuperResolutionModel):

    def __init__(self, scale_factor):
        super(DistilledResNetSR, self).__init__("DistilledResNetSR", scale_factor)

        # Treat this model as a denoising auto encoder
        # Force the fit, evaluate and upscale methods to take special care about image shape
        self.type_requires_divisible_shape = True
        self.uses_learning_phase = False

        self.n = 32
        self.mode = 2

        self.weight_path = "weights/DistilledResNetSR %dX.h5" % (self.scale_factor)
        self.type_true_upscaling = True

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        init =  super(DistilledResNetSR, self).create_model(height, width, channels, load_weights, batch_size)

        x0 = Convolution2D(self.n, (3, 3), activation='relu', padding='same', name='student_sr_res_conv1')(init)

        x = self._residual_block(x0, 1)

        x = Add(name='student_residual')([x, x0])
        x = self._upscale_block(x, 1)

        x = Convolution2D(3, (3, 3), activation="linear", padding='same', name='student_sr_res_conv_final')(x)

        model = Model(init, x)
        # dont compile yet
        if load_weights: model.load_weights(self.weight_path, by_name=True)

        self.model = model
        return model

    def _residual_block(self, ip, id):
        mode = False if self.mode == 2 else None
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        init = ip

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_1')(ip)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_1")(x, training=mode)
        
        x = Activation('relu', name="student_sr_res_activation_" + str(id) + "_1")(x)

        x = Convolution2D(self.n, (3, 3), activation='linear', padding='same',
                          name='student_sr_res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=channel_axis, name="student_sr_res_batchnorm_" + str(id) + "_2")(x, training=mode)

        m = Add(name="student_sr_res_merge_" + str(id))([x, init])

        return m

    def _upscale_block(self, ip, id):
        init = ip

        channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
        channels = init._keras_shape[channel_dim]

        x = UpSampling2D(name='student_upsampling_%d' % id)(init)
        x = Convolution2D(self.n * 2, (3, 3), activation="relu", padding='same', name='student_sr_res_filter1_%d' % id)(x)

        return x

    def fit(self, batch_size=128, nb_epochs=100, save_history=True, history_fn="Distilled ResNetSR History.txt"):
        super(DistilledResNetSR, self).fit(batch_size, nb_epochs, save_history, history_fn)