from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from lasagne.updates import nesterov_momentum

from helper import *

class CNN(NeuralNet):

    def __init__(
            self,
 
            **kwargs
    ):

        print kwargs

        NeuralNet.__init__(self, **kwargs,
                            update=nesterov_momentum,
                            update_learning_rate=0.001,
                            update_momentum=0.9,
                            # update_learning_rate=theano.shared(float32(0.03)),
                            # update_momentum=theano.shared(float32(0.9)),
                            loss=None,
                            objective=objective,
                            objective_loss_function=None,
                            batch_iterator_train=MyBatchIterator(batch_size=100),
                            batch_iterator_test=MyBatchIterator(batch_size=100),
                            regression=True,
                            max_epochs=300,
                            train_split=TrainSplit(eval_size=0.25),
                            custom_scores=None,
                            X_tensor_type=None, 
                            y_tensor_type=None,
                            use_label_encoder=False,
                            on_batch_finished=None,
                            on_epoch_finished=[
                                # AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
                                # AdjustVariable('update_momentum', start=0.9, stop=0.999),
                                EarlyStopping(patience=30),
                                ],
                            on_training_started=None,
                            on_training_finished=None,
                            more_params=None,
                            verbose=True)
                           # update,
                           # update_learning_rate,
                           # update_momentum,
                           # loss,
                           # objective,
                           # objective_loss_function,
                           # batch_iterator_train,
                           # batch_iterator_test,
                           # regression,
                           # max_epochs,
                           # train_split,
                           # custom_scores,
                           # X_tensor_type,
                           # y_tensor_type,
                           # use_label_encoder,
                           # on_batch_finished,
                           # on_epoch_finished,
                           # on_training_started,
                           # on_training_finished,
                           # more_params,
                           # verbose)
