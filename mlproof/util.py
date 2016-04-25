import glob
import mahotas as mh
import os

class Util(object):

  @staticmethod
  def read_section(path, z):
    '''
    '''
    image = glob.glob(os.path.join(path, 'image', '*'+str(z)+'.png'))    
    mask = glob.glob(os.path.join(path, 'mask', '*'+str(z)+'.png'))    
    gold = glob.glob(os.path.join(path, 'gold', '*'+str(z)+'.png'))    
    rhoana = glob.glob(os.path.join(path, 'rhoana', '*'+str(z)+'.png'))    
    prob = glob.glob(os.path.join(path, 'prob', '*'+str(z)+'.png'))    

    image = mh.imread(image[0])
    mask = mh.imread(mask[0])
    gold = mh.imread(gold[0])
    rhoana = mh.imread(rhoana[0])

    #convert ids from rgb to single channel
    rhoana_single = np.zeros(rhoana.shape, dtype=np.uint64)
    rhoana_single[:, :] = rhoana[:,:,0]*256*256 + rhoana[:,:,1]*256 + rhoana[:,:,2]
    gold_single = np.zeros(gold.shape, dtype=np.uint64)
    gold_single[:, :] = gold[:,:,0]*256*256 + gold[:,:,1]*256 + gold[:,:,2]


    #mask the rhoana output
    rhoana_single[mask==0] = 0


    return image, mask, gold_single, rhoana_single

      
