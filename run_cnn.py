import argparse

import mlproof as mlp
import mlproof.nets as nets

import os

import cPickle as picke


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
  OUTPUT_FOLDER = os.path.expanduser('~/nets/MergeNet_'+PATCH_PATH+'_'+BORDER+'_'+DESC)

  if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


  # create CNN
  cnn = eval("nets."+CNN_NAME)()

  # load data
  X_train, y_train, X_test, y_test = mlp.Patch.load(PATCH_PATH)

  # train
  cnn = cnn.fit({'image_input': X_train['image'],
                 'prob_input': X_train['prob'],
                 'binary_input': X_train['merged_array'],
                 'border_input': X_train[BORDER]},
                 y_train)

  # test
  test_accuracy = cnn.score({'image_input': X_test['image'],
                             'prob_input': X_test['prob'],
                             'binary_input': X_test['merged_array'],
                             'border_input': X_test[BORDER]},
                             y_test)

  print test_accuracy

  with open(os.path.join(OUTPUT_FOLDER, 'test_'+str(test_accuracy)+'.txt'), 'w') as f:
    f.write(str(test_accuracy))

  # store CNN
  with open(os.path.join(OUTPUT_FOLDER, 'net.p'), 'wb') as f:
    pickle.dump(cnn, f, -1)

  print 'All done.'
