import mlproof as mlp
import mlproof.nets as nets

import os

import cPickle as picke


CNN_NAME = 'MergeNet'
PATCH_PATH = 'cylinder_small1'
BORDER = 'larger_border_overlap'
DESC = 'test'
OUTPUT_FOLDER = os.path.expanduser('~/nets/MergeNet_'+BORDER+'_'+DESC)

if not os.path.exists(OUTPUT_FOLDER):
  os.makedirs(OUTPUT_FOLDER)


# create CNN
cnn = eval("nets."+CNN_NAME)

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

with open(os.path.join(OUTPUT_PATH, 'test_'+str(test_accuracy)+'.txt', 'w') as f:
  f.write(str(test_accuracy))

# store CNN
with open(os.path.join(OUTPUT_FOLDER, 'net.p'), 'wb') as f:
  pickle.dump(cnn, f, -1)

print 'All done.'
