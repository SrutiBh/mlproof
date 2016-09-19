import cPickle as pickle
import mahotas as mh
import numpy as np
import os
import time

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



PATCH_PATH = '/n/pfister_lab/haehn/patches/cylinder2_rgba/'#os.path.expanduser('~/patches_local/cylinder2_rgba_pickle/')

FILENAME = 'train_IPMB'


patch_file = 'correct_patches.p' ### <---- ADJUST

with open(PATCH_PATH+patch_file, 'rb') as f:
    patches = pickle.load(f)

print 'correct patches loaded'

NO_PATCHES = len(patches)*2 # two times for correct and error patches

PATCH_BYTES = 75*75
P_SIZE = (NO_PATCHES, 4, 75,75) ### <---- ADJUST

p_rgba = np.zeros(P_SIZE, dtype=np.float32)
p_target = np.zeros(NO_PATCHES)



i = 0
for p in patches:

        p_rgba[i][0] = p['image']
        p_rgba[i][1] = 1. - p['prob'] 
        p_rgba[i][2] = p['merged_array']
        p_rgba[i][3] = p['border_overlap']
        
        p_target[i] = 0 # <--- important
        i += 1


patch_file = 'error_patches.p' ### <---- ADJUST

with open(PATCH_PATH+patch_file, 'rb') as f:
    patches = pickle.load(f)

print 'error patches loaded'

for p in patches:

        p_rgba[i][0] = p['image']
        p_rgba[i][1] = 1. - p['prob'] 
        p_rgba[i][2] = p['merged_array']
        p_rgba[i][3] = p['border_overlap']
        
        p_target[i] = 1 # <--- important
        i += 1

#
# store unshuffled
#
np.savez(PATCH_PATH+FILENAME+'_unshuffled.npz', rgba=p_rgba)
np.savez(PATCH_PATH+FILENAME+'_targets_unshuffled.npz', targets=p_target)

print 'stored unshuffled.'

# now shuffle
shuffled = shuffle_in_unison_inplace(p_rgba, p_target)

np.savez(PATCH_PATH+FILENAME+'.npz', rgba=shuffled[0])
np.savez(PATCH_PATH+FILENAME+'_targets.npz', targets=shuffled[1])

print 'all done'
