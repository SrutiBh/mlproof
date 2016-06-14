from lasagne import layers
from lasagne import nonlinearities

from cnn import CNN

class MergeNetThreeLegDropout(CNN):
    '''
    Our CNN with different "legs" for image, prob, merged_array.

    Now with dropout layers. Three leg version!
    '''

    def __init__(self):

        CNN.__init__(self,

            layers=[
                ('image_input', layers.InputLayer),
                ('image_conv1', layers.Conv2DLayer),
                ('image_pool1', layers.MaxPool2DLayer),
                ('image_conv2', layers.Conv2DLayer),
                ('image_pool2', layers.MaxPool2DLayer),
                ('image_dropout', layers.DropoutLayer),

                ('prob_input', layers.InputLayer),
                ('prob_conv1', layers.Conv2DLayer),
                ('prob_pool1', layers.MaxPool2DLayer),
                ('prob_conv2', layers.Conv2DLayer),
                ('prob_pool2', layers.MaxPool2DLayer),
                ('prob_dropout', layers.DropoutLayer),

                ('binary_input', layers.InputLayer),
                ('binary_conv1', layers.Conv2DLayer),
                ('binary_pool1', layers.MaxPool2DLayer),
                ('binary_conv2', layers.Conv2DLayer),
                ('binary_pool2', layers.MaxPool2DLayer),
                ('binary_dropout', layers.DropoutLayer),

                ('merge', layers.ConcatLayer),
                ('hidden3', layers.DenseLayer),
                ('dropout3', layers.DropoutLayer),
                ('output', layers.DenseLayer),
            ],

            # input
            image_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            image_conv1_filter_size=(13,13), image_conv1_num_filters=24,
            image_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            image_conv2_filter_size=(13,13), image_conv2_num_filters=22,
            image_pool2_pool_size=(2,2),
            image_dropout_p=0.2,

            prob_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            prob_conv1_filter_size=(13,13), prob_conv1_num_filters=24,
            prob_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            prob_conv2_filter_size=(13,13), prob_conv2_num_filters=22,
            prob_pool2_pool_size=(2,2),
            prob_dropout_p=0.2,

            binary_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            binary_conv1_filter_size=(13,13), binary_conv1_num_filters=24,
            binary_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            binary_conv2_filter_size=(13,13), binary_conv2_num_filters=22,
            binary_pool2_pool_size=(2,2),
            binary_dropout_p=0.2,

            # concat
            # merge_incomings=['image_pool2','prob_pool2','binary_pool2','border_pool2'],
            merge_incomings=['image_dropout','prob_dropout','binary_dropout'],

            # dense layer 1
            hidden3_num_units=256,
            hidden3_nonlinearity=nonlinearities.rectify,
            dropout3_p=0.5,

            # dense layer 2
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax

        )
