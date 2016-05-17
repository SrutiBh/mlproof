from lasagne import layers
from lasagne import nonlinearities

from cnn import CNN

class MergeNet3(CNN):
    '''
    Our CNN with different "legs" for image, prob, merged_array, border_overlap.

    And with 3 layers.
    '''

    def __init__(self):

        CNN.__init__(self,

            layers=[
                ('image_input', layers.InputLayer),
                ('image_conv1', layers.Conv2DLayer),
                ('image_pool1', layers.MaxPool2DLayer),
                ('image_conv2', layers.Conv2DLayer),
                ('image_pool2', layers.MaxPool2DLayer),
                ('image_conv3', layers.Conv2DLayer),
                ('image_pool3', layers.MaxPool2DLayer),

                ('prob_input', layers.InputLayer),
                ('prob_conv1', layers.Conv2DLayer),
                ('prob_pool1', layers.MaxPool2DLayer),
                ('prob_conv2', layers.Conv2DLayer),
                ('prob_pool2', layers.MaxPool2DLayer),
                ('prob_conv3', layers.Conv2DLayer),
                ('prob_pool3', layers.MaxPool2DLayer),

                ('binary_input', layers.InputLayer),
                ('binary_conv1', layers.Conv2DLayer),
                ('binary_pool1', layers.MaxPool2DLayer),
                ('binary_conv2', layers.Conv2DLayer),
                ('binary_pool2', layers.MaxPool2DLayer),
                ('binary_conv3', layers.Conv2DLayer),
                ('binary_pool3', layers.MaxPool2DLayer),

                ('border_input', layers.InputLayer),
                ('border_conv1', layers.Conv2DLayer),
                ('border_pool1', layers.MaxPool2DLayer),
                ('border_conv2', layers.Conv2DLayer),
                ('border_pool2', layers.MaxPool2DLayer),                                
                ('border_conv3', layers.Conv2DLayer),
                ('border_pool3', layers.MaxPool2DLayer),

                ('merge', layers.ConcatLayer),
                ('hidden3', layers.DenseLayer),
                ('dropout3', layers.DropoutLayer),
                ('output', layers.DenseLayer),
            ],

            # input
            image_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            image_conv1_filter_size=(13,13), image_conv1_num_filters=16,
            image_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            image_conv2_filter_size=(13,13), image_conv2_num_filters=16,
            image_pool2_pool_size=(2,2),
            image_conv3_filter_size=(7,7), image_conv3_num_filters=16,
            image_pool3_pool_size=(2,2),

            prob_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            prob_conv1_filter_size=(13,13), prob_conv1_num_filters=16,
            prob_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            prob_conv2_filter_size=(13,13), prob_conv2_num_filters=16,
            prob_pool2_pool_size=(2,2),
            prob_conv3_filter_size=(7,7), prob_conv3_num_filters=16,
            prob_pool3_pool_size=(2,2),            

            binary_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            binary_conv1_filter_size=(13,13), binary_conv1_num_filters=16,
            binary_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            binary_conv2_filter_size=(13,13), binary_conv2_num_filters=16,
            binary_pool2_pool_size=(2,2),
            binary_conv3_filter_size=(7,7), binary_conv3_num_filters=16,
            binary_pool3_pool_size=(2,2),            

            border_input_shape=(None, 1, 75, 75),
            # conv2d + pool + dropout
            border_conv1_filter_size=(13,13), border_conv1_num_filters=16,
            border_pool1_pool_size=(2,2),
            # conv2d + pool + dropout
            border_conv2_filter_size=(13,13), border_conv2_num_filters=16,
            border_pool2_pool_size=(2,2),
            border_conv3_filter_size=(7,7), border_conv3_num_filters=16,
            border_pool3_pool_size=(2,2),            

            # concat
            merge_incomings=['image_pool3','prob_pool3','binary_pool3','border_pool3'],

            # dense layer 1
            hidden3_num_units=256,
            hidden3_nonlinearity=nonlinearities.rectify,
            dropout3_p=0.5,

            # dense layer 2
            output_num_units=2,
            output_nonlinearity=nonlinearities.softmax

        )
