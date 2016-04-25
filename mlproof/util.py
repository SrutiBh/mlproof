import glob
import mahotas as mh
import numpy as np
import os

class Util(object):

  @staticmethod
  def read_section(path, z):
    '''
    '''
    image = sorted(glob.glob(os.path.join(path, 'image', '*'+str(z)+'.png')))
    mask = sorted(glob.glob(os.path.join(path, 'mask', '*'+str(z)+'.png')))   
    gold = sorted(glob.glob(os.path.join(path, 'gold', '*'+str(z)+'.png')))
    rhoana = sorted(glob.glob(os.path.join(path, 'rhoana', '*'+str(z)+'.png')))
    prob = sorted(glob.glob(os.path.join(path, 'prob', '*'+str(z)+'.png')))

    print 'Loading', image[0]

    image = mh.imread(image[0])
    mask = mh.imread(mask[0])
    gold = mh.imread(gold[0])
    rhoana = mh.imread(rhoana[0])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros((rhoana.shape[0], rhoana.shape[1]), dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]
    gold_single = np.zeros((gold.shape[0], gold.shape[1]), dtype=np.uint64)
    gold_single[:, :] = gold[:,:,0]*256*256 + gold[:,:,1]*256 + gold[:,:,2]


    #mask the rhoana output
    rhoana_single[mask==0] = 0


    return image, mask, gold_single, rhoana_single

      
