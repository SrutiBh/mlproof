import collections
import cPickle as pickle
import lasagne
import numpy as np
import os

from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

from sklearn.metrics import classification_report, accuracy_score

import mlproof as mlp


#
# we need to overload our batch iterator
#

from nolearn.lasagne import BatchIterator

class MyTestBatchIterator(BatchIterator):
    def transform(self, Xb, yb):

        # regularize the batch (which is already in the range 0..1)
        if isinstance(Xb, dict):
            # this is for our multi-leg CNN

            for k in Xb:
                Xb[k] = (Xb[k] - .5).astype(np.float32)

        else:

            Xb = Xb - .5
            
        return Xb, yb
#
#
#


class Stats(object):

  @staticmethod
  def load_cnn(path):

    # load cnn
    with open(path, 'rb') as f:
      cnn = pickle.load(f)    

    # make sure we have the correct test batch iterator
    cnn.batch_iterator_test = MyTestBatchIterator(100)


    # load patches
    X_train, y_train, X_test, y_test = mlp.Patch.load('cylinder1')

    test_inputs = collections.OrderedDict()
    input_names = []
    input_values = []
    for l in cnn.layers:
      layer_name, layer_type = l
      if layer_type == lasagne.layers.input.InputLayer:
        input_name = layer_name.split('_')[0]
        if input_name == 'binary':
          input_name = 'merged_array'
        if input_name == 'border':
          input_name = 'border_overlap'
          if path.find('larger_border_overlap') != -1:
            input_name = 'larger_border_overlap'

        input_names.append(layer_name)
        input_values.append(input_name)
        test_inputs[layer_name] = X_test[input_name]

    print 'Using test set:', input_values

    # calc F1
    test_prediction = cnn.predict(test_inputs)
    print
    print 'Precision/Recall:'
    print classification_report(y_test, test_prediction)

    # calc test accuracy
    test_acc = cnn.score(test_inputs, y_test)
    acc_score = accuracy_score(y_test, test_prediction)
    print 'Test Accuracy:', test_acc
    print 'Accuracy Score:', acc_score

    # plot loss
    loss_plot = plot_loss(cnn)
    # loss_plot.savefig('/tmp/aaa.png')

    # attach patch selection
    cnn.input_names = input_names
    cnn.input_values = input_values
    cnn.uuid = os.path.basename(os.path.dirname(path))

    return cnn, loss_plot

  @staticmethod
  def run_dojo_xp(cnn):

    # load dojo data
    input_image, input_prob, input_gold, input_rhoana, dojo_bbox = mlp.Legacy.read_dojo_data()


    original_mean_VI, original_median_VI, original_VI_s = mlp.Legacy.VI(input_gold, input_rhoana)

    # output folder for anything to store
    output_folder = '/tmp/netstats/'+cnn.uuid+'/'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    # find merge errors, if we did not generate them before
    merge_error_file = output_folder+'/merge_errors.p'
    if os.path.exists(merge_error_file):
      print 'Loading merge errors from file..'
      with open(merge_error_file, 'rb') as f:
        merge_errors = pickle.load(f)
    else:
      print 'Finding Top 5 merge errors..'
      merge_errors = mlp.Legacy.get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana)
      with open(merge_error_file, 'wb') as f:
        pickle.dump(merge_errors, f)

    print len(merge_errors), ' merge errors found.'



    # print
    # #
    # # perform merge correction with p < .05
    # #

    # print 'Correcting merge errors with p < .05'
    # corrected_rhoana_05 = mlp.Legacy.perform_auto_merge_correction(input_image, input_rhoana, merge_errors, .05)

    # print '   Mean VI improvement', original_mean_VI-mlp.Legacy.VI(input_gold, corrected_rhoana_05)[0]

    # #
    # # perform split correction with p > .95
    # #
    # print 'Correcting split errors with p > .95'
    # vi_95 = mlp.Legacy.perform_auto_split_correction(cnn, input_image, input_prob, corrected_rhoana_05, input_gold, .95)

    # print '   Mean VI improvement', original_mean_VI-vi_95[0]




    # print
    # #
    # # perform merge correction with p < .01
    # #

    # print 'Correcting merge errors with p < .01'
    # corrected_rhoana_01 = mlp.Legacy.perform_auto_merge_correction(input_image, input_rhoana, merge_errors, .01)

    # print '   Mean VI improvement', original_mean_VI-mlp.Legacy.VI(input_gold, corrected_rhoana_01)[0]

    # #
    # # perform split correction with p > .99
    # #
    # print 'Correcting split errors with p > .99'
    # vi_99 = mlp.Legacy.perform_auto_split_correction(cnn, input_image, input_prob, corrected_rhoana_01, input_gold, .99)

    # print '   Mean VI improvement', original_mean_VI-vi_99[0]


    print
    #
    # perform merge correction with simulated user
    #
    print 'Correcting merge errors by simulated user (er=0)'
    corrected_rhoana_sim_user, sim_user_fixes = mlp.Legacy.perform_sim_user_merge_correction(input_image, input_gold, input_rhoana, merge_errors)
    
    print '   Mean VI improvement', original_mean_VI-mlp.Legacy.VI(input_gold, corrected_rhoana_sim_user)[0]
    
    #
    # perform split correction with simulated user
    #

    # we need to create a bigM for the dojo volume
    bigM_dojo_file = output_folder + '/bigM_dojo.p'
    if os.path.exists(bigM_dojo_file):
      print 'Loading dojo bigM from file..'
      with open(bigM_dojo_file, 'rb') as f:
        bigM_dojo = pickle.load(f)
    else:
      print 'Creating dojo bigM..'
      bigM_dojo = mlp.Legacy.create_bigM_without_mask(cnn, input_image, input_prob, input_rhoana, verbose=False)
      with open(bigM_dojo_file, 'wb') as f:
        pickle.dump(bigM_dojo, f)    


    return bigM_dojo
    





    # perform merge correction with p < .01
    return merge_errors

    # 

