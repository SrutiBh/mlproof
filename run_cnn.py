import argparse

import mlproof as mlp
import mlproof.nets as nets

import numpy as np

import os

import cPickle as pickle

import sys
sys.setrecursionlimit(1000000000)


if __name__ == '__main__':


  parser = argparse.ArgumentParser()
  parser.add_argument("-n", "--name", type=str, help="the cnn type", default='MergeNet')
  parser.add_argument("-p", "--patchpath", type=str, help="the patch folder in the datapath", default='cylinder_small1')
  parser.add_argument("-b", "--border", type=str, help="the border to use", default='larger_border_overlap')
  parser.add_argument("-d", "--desc", type=str, help="the description", default='test')

  args = parser.parse_args()

  CNN_NAME = args.name
  PATCH_PATH = args.patchpath
  BORDER = args.border
  DESC = args.desc
  OUTPUT_FOLDER = os.path.expanduser('~/nets/'+CNN_NAME+'_'+PATCH_PATH+'_'+BORDER+'_'+DESC)

  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


  # create CNN
  cnn = eval("nets."+CNN_NAME)()

  # load data
  X_train, y_train, X_test, y_test = mlp.Patch.load(PATCH_PATH)

  #
  # train and test inputs
  #
  if CNN_NAME.startswith('MergeNet'):
    # 4 leg version
    X_train_input = {'image_input': X_train['image'],
                     'prob_input': X_train['prob'],
                     'binary_input': X_train['merged_array'],
                     'border_input': X_train[BORDER]}

    X_test_input = {'image_input': X_test['image'],
                    'prob_input': X_test['prob'],
                    'binary_input': X_test['merged_array'],
                    'border_input': X_test[BORDER]}

  elif CNN_NAME.startswith('RGBANet'):
    # rgba version
    X_train_input = np.concatenate((X_train['image'], 
                                    X_train['prob'], 
                                    X_train['merged_array'],
                                    X_train[BORDER]), 1)

    X_test_input = np.concatenate((X_test['image'], 
                                  X_test['prob'], 
                                  X_test['merged_array'],
                                  X_test[BORDER]), 1)



  # train
  cnn = cnn.fit(X_train_input, y_train)

  # test
  test_accuracy = cnn.score(X_test_input, y_test)

  print test_accuracy

  with open(os.path.join(OUTPUT_FOLDER, 'test_'+str(test_accuracy)+'.txt'), 'w') as f:
    f.write(str(test_accuracy))

  # store CNN
  with open(os.path.join(OUTPUT_FOLDER, 'net.p'), 'wb') as f:
    pickle.dump(cnn, f, -1)

  print 'All done.'
