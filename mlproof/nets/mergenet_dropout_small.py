from lasagne import layers
from lasagne import nonlinearities

from cnn import CNN

class MergeNetDropoutSmall(CNN):
    '''
    Our CNN with different "legs" for image, prob, merged_array, border_overlap.

    Now with dropout layers.
    '''

    def __init__(self):

        CNN.__init__(self,

            layers=[
                ('image_input', layers.InputLayer),
                ('image_conv1', layers.Conv2DLayer),
                ('image_pool1', layers.MaxPool2DLayer),
                ('image_dropout1', layers.DropoutLayer),
                ('image_conv2', layers.Conv2DLayer),
                ('image_pool2', layers.MaxPool2DLayer),
                ('image_dropout2', layers.DropoutLayer),

                ('prob_input', layers.InputLayer),
                ('prob_conv1', layers.Conv2DLayer),
                ('prob_pool1', layers.MaxPool2DLayer),
                ('prob_dropout1', layers.DropoutLayer),
                ('prob_conv2', layers.Conv2DLayer),
                ('prob_pool2', layers.MaxPool2DLayer),
                ('prob_dropout2', layers.DropoutLayer),

                ('binary_input', layers.InputLayer),
                ('binary_conv1', layers.Conv2DLayer),
                ('binary_pool1', layers.MaxPool2DLayer),
                ('binary_dropout1', layers.DropoutLayer),
                ('binary_conv2', layers.Conv2DLayer),
                ('binary_pool2', layers.MaxPool2DLayer),
                ('binary_dropout2', layers.DropoutLayer),

                ('border_input', layers.InputLayer),
                ('border_conv1', layers.Conv2DLayer),
                ('border_pool1', layers.MaxPool2DLayer),
                ('border_dropout1', layers.DropoutLayer),
                ('border_conv2', layers.Conv2DLayer),
                ('border_pool2', layers.MaxPool2DLayer),                                
                ('border_dropout2', layers.DropoutLayer),

                ('merge', layers.ConcatLayer),
                ('hidden3', layers.DenseLayer),
                ('dropout3', layers.DropoutLayer),
                ('output', layers.DenseLayer),
            ],

            # input
            image_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            image_conv1_filter_size=(3,3), image_conv1_num_filters=64,
            image_pool1_pool_size=(2,2),
            image_dropout1_p=0.2,
            # conv2d + pool + dropout
            image_conv2_filter_size=(3,3), image_conv2_num_filters=64,
            image_pool2_pool_size=(2,2),
            image_dropout2_p=0.2,

            prob_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            prob_conv1_filter_size=(3,3), prob_conv1_num_filters=64,
            prob_pool1_pool_size=(2,2),
            prob_dropout1_p=0.2,
            # conv2d + pool + dropout
            prob_conv2_filter_size=(3,3), prob_conv2_num_filters=64,
            prob_pool2_pool_size=(2,2),
            prob_dropout2_p=0.2,

            binary_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            binary_conv1_filter_size=(3,3), binary_conv1_num_filters=64,
            binary_pool1_pool_size=(2,2),
            binary_dropout1_p=0.2,
            # conv2d + pool + dropout
            binary_conv2_filter_size=(3,3), binary_conv2_num_filters=64,
            binary_pool2_pool_size=(2,2),
            binary_dropout2_p=0.2,

            border_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            border_conv1_filter_size=(3,3), border_conv1_num_filters=64,
            border_pool1_pool_size=(2,2),
            border_dropout1_p=0.2,
            # conv2d + pool + dropout
            border_conv2_filter_size=(3,3), border_conv2_num_filters=64,
            border_pool2_pool_size=(2,2),
            border_dropout2_p=0.2,

            # concat
            # merge_incomings=['image_pool2','prob_pool2','binary_pool2','border_pool2'],
            merge_incomings=['image_dropout2','prob_dropout2','binary_dropout2','border_dropout2'],

            # dense layer 1
            hidden3_num_units=512,
            hidden3_nonlinearity=nonlinearities.rectify,
            dropout3_p=0.5,

            # dense layer 2
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax

        )
