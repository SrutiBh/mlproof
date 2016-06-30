import cPickle as pickle
import mahotas as mh
import numpy as np
import os
import time
import sys

import mlproof as mlp

PATCH_PATH = os.path.expanduser('~/patches/cylinder1_rgba/')


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle_patches(outname, borderprefix):

  patches = np.load(PATCH_PATH+outname+'_'+borderprefix+'_unshuffled.npz')
  targets = np.load(PATCH_PATH+outname+'_'+borderprefix+'_targets_unshuffled.npz')

  shuffled = shuffle_in_unison_inplace(patches, targets)

  print 'saving'
  np.savez(PATCH_PATH+outname+'_'+borderprefix+'.npz', rgba=shuffled[0])
  np.savez(PATCH_PATH+outname+'_'+borderprefix+'_targets.npz', rgba=shuffled[1])
  print 'Done!'


shuffle_patches(sys.argv[1], sys.argv[2])
