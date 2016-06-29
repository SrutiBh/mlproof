import cPickle as pickle
import mahotas as mh
import numpy as np
import os
import time
import sys

import mlproof as mlp

PATCH_PATH = os.path.expanduser('~/patches/cylinder1_rgba/')
OUTPUT_PATH = os.path.expanduser('~/patches/cylinder1_rgba')


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def run(outname, borderprefix):

  NO_PATCHES = 13408*2#266088*2

  PATCH_BYTES = 75*75
  P_SIZE = (NO_PATCHES, 4, 75,75) # rather than raveled right now
  
  p_rgba = np.zeros(P_SIZE, dtype=np.float32)
  
  p_target = np.zeros(NO_PATCHES)

  i = 0

  groups = [(outname,200,250)]

  for g in groups:

    with open(PATCH_PATH+g[0]+'_'+str(g[1])+'_'+str(g[2])+'_error_patches.p', 'rb') as f:
      patches = pickle.load(f)

      for p in patches:

          p_rgba[i][0] = p['image']
          p_rgba[i][1] = p['prob'] 
          p_rgba[i][2] = p['merged_array']
          p_rgba[i][3] = p[borderprefix+'_overlap']
          
          p_target[i] = 1 # <--- important
          i += 1

    with open(PATCH_PATH+g[0]+'_'+str(g[1])+'_'+str(g[2])+'_correct_patches.p', 'rb') as f:
      patches = pickle.load(f)

      for p in patches:

          p_rgba[i][0] = p['image']
          p_rgba[i][1] = p['prob'] 
          p_rgba[i][2] = p['merged_array']
          p_rgba[i][3] = p[borderprefix+'_overlap']
          
          p_target[i] = 1 # <--- important
          i += 1

  shuffled = shuffle_in_unison_inplace(p_rgba, p_target)

  print 'saving'
  np.savez(PATCH_PATH+groups[0][0]+'_'+borderprefix+'.npz', rgba=shuffled[0])
  np.savez(PATCH_PATH+groups[0][0]+'_'+borderprefix+'_targets.npz', rgba=shuffled[1])
  print 'Done!'

run(sys.argv[1], sys.argv[2])