import numpy as np
import mahotas as mh
import random
from scipy.spatial import distance
import skimage.measure
import tifffile as tif
import time

from patch import Patch
from util import Util

class Legacy(object):

  @staticmethod
  def read_dojo_data():
    input_image = np.zeros((10,1024,1024))
    input_rhoana = np.zeros((10,1024,1024))
    input_gold = np.zeros((10,1024,1024))
    input_prob = np.zeros((10,1024,1024))
    path_prefix = '/Users/d/Projects/'
    path_prefix = '/home/d/dojo_xp/data/' # for beast only
    input_rhoana = tif.imread(path_prefix+'dojo_data_vis2014/labels_after_automatic_segmentation_multi.tif')
    input_gold = tif.imread(path_prefix+'dojo_data_vis2014/groundtruth_multi.tif')
    for i in range(10):
        input_prob[i] = tif.imread(path_prefix+'dojo_data_vis2014/prob/'+str(i)+'_syn.tif')
        input_image[i] = tif.imread(path_prefix+'dojo_data_vis2014/images/'+str(i)+'.tif')
        
    bbox = mh.bbox(input_image[0])
    bbox_larger = [bbox[0]-37, bbox[1]+37, bbox[2]-37, bbox[3]+37]

    prob_new = np.zeros(input_image.shape, dtype=np.uint8)
    
    input_image = input_image[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_rhoana = input_rhoana[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    input_gold = input_gold[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    # input_prob = input_prob[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]
    
    prob_new[:,bbox[0]:bbox[1], bbox[2]:bbox[3]] = input_prob[:,bbox[0]:bbox[1], bbox[2]:bbox[3]]
    prob_new = prob_new[:, bbox_larger[0]:bbox_larger[1], bbox_larger[2]:bbox_larger[3]]



    for i in range(0,10):
      zeros_gold = Util.threshold(input_gold[i], 0)
      input_gold[i] = Util.relabel(input_gold[i])
      # restore zeros
      input_gold[i][zeros_gold==1] = 0
      input_rhoana[i] = Util.relabel(input_rhoana[i])

    return input_image.astype(np.uint8), prob_new.astype(np.uint8), input_gold.astype(np.uint32), input_rhoana.astype(np.uint32), bbox_larger

  @staticmethod
  def invert(array, smooth=False, sigma=2.5):
    
    grad = mh.gaussian_filter(array, sigma)

    return (255-grad)

  @staticmethod
  def gradient(array, sigma=2.5):
    '''
    '''

    grad = mh.gaussian_filter(array, sigma)

    grad_x = np.gradient(grad)[0]
    grad_y = np.gradient(grad)[1]
    grad = np.sqrt(np.add(grad_x*grad_x, grad_y*grad_y))

    grad -= grad.min()
    grad /= (grad.max() - grad.min())
    grad *= 255

    return grad


  @staticmethod
  def random_watershed(array, speed_image, border_seeds=False, erode=False):
    '''
    '''
    copy_array = np.array(array, dtype=np.bool)

    if erode:
      
      for i in range(10):
        copy_array = mh.erode(copy_array)


    seed_array = np.array(copy_array)
    if border_seeds:
      seed_array = mh.labeled.border(copy_array, 1, 0, Bc=mh.disk(7))

    coords = zip(*np.where(seed_array==1))

    if len(coords) == 0:
      # print 'err'
      return np.zeros(array.shape)

    seed1_ = None
    seed2_ = None
    max_distance = -np.inf

    for i in range(10):
      seed1 = random.choice(coords)
      seed2 = random.choice(coords)
      d = distance.euclidean(seed1, seed2)
      if max_distance < d:
        max_distance = d
        seed1_ = seed1
        seed2_ = seed2

    seeds = np.zeros(array.shape, dtype=np.uint8)
    seeds[seed1_[0], seed1_[1]] = 1
    seeds[seed2_[0], seed2_[1]] = 2



    for i in range(8):
      seeds = mh.dilate(seeds)

    # Util.view(seeds,large=True)      
    # print speed_image.shape, seeds.shape
    ws = mh.cwatershed(speed_image, seeds)
    ws[array == 0] = 0

    return ws    


  @staticmethod
  def fix_single_merge(cnn, cropped_image, cropped_prob, cropped_binary, N=10, invert=True, dilate=True, 
                       border_seeds=True, erode=False, debug=False, before_merge_error=None,
                       real_border=np.zeros((1,1)), oversampling=False, crop=True):
    '''
    invert: True/False for invert or gradient image
    '''

    bbox = mh.bbox(cropped_binary)

    orig_cropped_image = np.array(cropped_image)
    orig_cropped_prob  = np.array(cropped_prob)
    orig_cropped_binary = np.array(cropped_binary)



    speed_image = None
    if invert:
      speed_image = Legacy.invert(cropped_image, smooth=True, sigma=2.5)
    else:
      speed_image = Legacy.gradient(cropped_image)


    dilated_binary = np.array(cropped_binary, dtype=np.bool)
    if dilate:
      for i in range(20):
          dilated_binary = mh.dilate(dilated_binary)      

    # Util.view(dilated_binary, large=True)

    borders = np.zeros(cropped_binary.shape)

    best_border_prediction = np.inf
    best_border_image = np.zeros(cropped_binary.shape)

    original_border = mh.labeled.border(cropped_binary, 1, 0, Bc=mh.disk(3))

    results_no_border = []
    predictions = []
    borders = []
    results = []

    for n in range(N):
        ws = Legacy.random_watershed(dilated_binary, speed_image, border_seeds=border_seeds, erode=erode)
        
        if ws.max() == 0:
          continue

        ws_label1 = ws.max()
        ws_label2 = ws.max()-1
        border = mh.labeled.border(ws, ws_label1, ws_label2)

        # Util.view(ws, large=True)


        # Util.view(border, large=True)

        # print i, len(border[border==True])

        #
        # remove parts of the border which overlap with the original border
        #

        

        ws[cropped_binary == 0] = 0

        # Util.view(ws, large=False, color=False)        

        ws_label1_array = Util.threshold(ws, ws_label1)
        ws_label2_array = Util.threshold(ws, ws_label2)

        eroded_ws1 = np.array(ws_label1_array, dtype=np.bool)
        eroded_ws2 = np.array(ws_label2_array, dtype=np.bool)
        if erode:

          for i in range(5):
            eroded_ws1 = mh.erode(eroded_ws1)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws1 = np.array(eroded_ws1)
          for i in range(5):
            dilated_ws1 = mh.dilate(dilated_ws1)


          for i in range(5):
            eroded_ws2 = mh.erode(eroded_ws2)

          # Util.view(eroded_ws, large=True, color=False)

          dilated_ws2 = np.array(eroded_ws2)
          for i in range(5):
            dilated_ws2 = mh.dilate(dilated_ws2)




          new_ws = np.zeros(ws.shape, dtype=np.uint8)
          new_ws[dilated_ws1 == 1] = ws_label1
          new_ws[dilated_ws2 == 1] = ws_label2


          ws = new_ws

          # Util.view(new_ws, large=True, color=True)

        # ws[original_border == 1] = 0
        
        prediction = Patch.grab_group_test_and_unify(cnn, cropped_image, cropped_prob, ws, ws_label1, ws_label2, oversampling=oversampling)
        
        if prediction == -1 or prediction >= .5:
          # invalid
          continue


        # here we have for one border
        # the border
        # the prediction
        # borders.append(border)
        # predictions.append(prediction)
        results.append((prediction, border))



    return results




  @staticmethod
  def get_top5_merge_errors(cnn, input_image, input_prob, input_rhoana, verbose=False):

    #
    # this creates the top bins for the best five merge splits
    #
    t0 = time.time()
    fixed_volume = np.array(input_rhoana)

    merge_errors = []

    for i in range(10):
        if verbose:
          print 'working on slice', i
        
        DOJO_SLICE = i
        
        hist = Util.get_histogram(input_rhoana[DOJO_SLICE].astype(np.uint64))
        labels = range(len(hist))

        fixed_slice = np.array(input_rhoana[DOJO_SLICE], dtype=np.uint64)

        for l in labels:

            if l == 0 or hist[l]<3000:
                continue

            # single binary mask for label l
            before_merge_error = np.zeros(input_rhoana[DOJO_SLICE].shape)
            before_merge_error[fixed_slice == l] = 1

            results = Legacy.fix_single_merge(cnn,
                                              input_image[DOJO_SLICE],
                                              input_prob[DOJO_SLICE],
                                              before_merge_error, N=50, 
                                              erode=True, 
                                              invert=True,
                                              dilate=True,
                                              border_seeds=True,
                                              oversampling=False)

            if len(results) > 0:
                
                #
                # SORT THE PREDICTIONS (prediction, border)-tupels
                # LOOK AT TOP 5
                sorted_pred = sorted(results, key=lambda x: x[0])

                top5 = sorted_pred[:5]
                
                lowest_prediction = sorted_pred[0][0]
                

                # store the merge error
                # we need to store: z, l, results_no_border, borders, predictions
                merge_errors.append((i, l, lowest_prediction, (top5)))
                
    if verbose:
      print 'merge error correction done after',time.time()-t0, 'seconds'

    return merge_errors



  @staticmethod
  def get_merge_error_image(input_image, input_rhoana, label, border):

    binary = Util.threshold(input_rhoana, label)
    binary_dilated = mh.dilate(binary.astype(np.bool))
    for dilate in range(30):
      binary_dilated = mh.dilate(binary_dilated)


    binary_bbox = mh.bbox(binary_dilated)
    binary_border = mh.labeled.borders(binary)

    b = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    b[:,:,0] = input_image[:]
    b[:,:,1] = input_image[:]
    b[:,:,2] = input_image[:]
    b[:,:,3] = 255

    c = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    c[:,:,0] = input_image[:]
    c[:,:,1] = input_image[:]
    c[:,:,2] = input_image[:]
    c[:,:,3] = 255        
    c[binary_border == 1] = (0,255,0,255)

    e = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    e[:,:,0] = input_image[:]
    e[:,:,1] = input_image[:]
    e[:,:,2] = input_image[:]
    e[:,:,3] = 255        

    f = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    f[:,:,0] = input_image[:]
    f[:,:,1] = input_image[:]
    f[:,:,2] = input_image[:]
    f[:,:,3] = 255  
    f[binary == 1] = (0,255,0,255)

    g = np.zeros((input_image.shape[0],input_image.shape[1],4), dtype=np.uint8)
    g[:,:,0] = input_image[:]
    g[:,:,1] = input_image[:]
    g[:,:,2] = input_image[:]
    g[:,:,3] = 255  

    border[binary==0] = 0

    b[border == 1] = (255,0,0,255)
    b[binary_border == 1] = (0,255,0,255)

    cropped_image = Util.crop_by_bbox(input_image, binary_bbox)
    cropped_binary_border = Util.crop_by_bbox(c, binary_bbox)
    cropped_combined_border = Util.crop_by_bbox(b, binary_bbox)
    cropped_border_only = Util.crop_by_bbox(border, binary_bbox)

    corrected_binary = Legacy.correct_merge(input_rhoana, label, border)
    corrected_binary_original = np.array(corrected_binary)
    result = np.array(input_rhoana)
    corrected_binary += result.max()
    corrected_binary[corrected_binary_original == 0] = 0

    result[corrected_binary != 0] = 0
    result += corrected_binary.astype(np.uint64)
    cropped_result = Util.crop_by_bbox(corrected_binary, binary_bbox)

    g[corrected_binary_original==2] = (255,0,0,255)
    g[corrected_binary_original==1] = (0,255,0,255)
    cropped_fusion = Util.crop_by_bbox(g, binary_bbox)

    e[binary_bbox[0]:binary_bbox[1], binary_bbox[2]] = (255,255,0,255)
    e[binary_bbox[0]:binary_bbox[1], binary_bbox[3]] = (255,255,0,255)
    e[binary_bbox[0], binary_bbox[2]:binary_bbox[3]] = (255,255,0,255)
    e[binary_bbox[1], binary_bbox[2]:binary_bbox[3]] = (255,255,0,255)  

    sliceoverview = e

    cropped_binary = Util.crop_by_bbox(f, binary_bbox)

    return cropped_image, cropped_binary_border, cropped_combined_border, cropped_border_only, cropped_result, result, sliceoverview, cropped_binary, cropped_fusion

  @staticmethod
  def remove_border_mess(e):
    '''
    '''
    label_sizes = Util.get_histogram(e)
    # we only want to keep the two largest labels
    largest1 = np.argmax(label_sizes[1:])+1
    label_sizes[largest1] = 0
    largest2 = np.argmax(label_sizes[1:])+1
    label_sizes[largest2] = 0
    for l,s in enumerate(label_sizes):
        if l == 0 or s == 0:
            # this label has zero pixels anyways or is the background
            continue
        
        # find neighbor for l
        neighbors = Util.grab_neighbors(e, l)

        if largest1 in neighbors:
            # prefer the largest
            e[e==l] = largest1
        elif largest2 in neighbors:
            e[e==l] = largest2

    return e

  @staticmethod
  def correct_merge(input_rhoana, label, border):
    
    rhoana_copy = np.array(input_rhoana, dtype=np.uint64)

    # split the label using the border
    binary = Util.threshold(input_rhoana, label).astype(np.uint64)

    border[binary==0] = 0
    binary[border==1] = 2

    binary_relabeled = Util.relabel(binary)

    binary_no_border = np.array(binary_relabeled, dtype=np.uint64)
    binary_no_border[border==1] = 0
    

    sizes = mh.labeled.labeled_size(binary_no_border)
    too_small = np.where(sizes < 200)
    labeled_small = mh.labeled.remove_regions(binary_no_border, too_small)
    labeled_small_zeros = Util.threshold(labeled_small, 0)
    labeled_small = Util.fill(labeled_small, labeled_small_zeros.astype(np.bool))
    binary_no_border = Util.frame_image(labeled_small).astype(np.uint64)     
    binary_no_border[binary==0] = 0

    corrected_binary = binary_no_border

    # now let's remove the possible border mess
    n = 0
    while corrected_binary.max() != 2 and n < 6:
      corrected_binary = Legacy.remove_border_mess(corrected_binary)
      corrected_binary = skimage.measure.label(corrected_binary)
      n += 1

    return corrected_binary

  @staticmethod
  def perform_auto_merge_correction(input_image, input_rhoana, merge_errors, p):
    rhoana_after_merge_correction = np.array(input_rhoana)
    
    for me in merge_errors:
        pred = me[2]
        if pred < p:
            # print 'fixing', pred
            z = me[0]
            label = me[1]
            border = me[3][0][1]
            a,b,c,d,e,f,g,h,i = Legacy.get_merge_error_image(input_image[z], rhoana_after_merge_correction[z], label, border)        
            rhoana_after_merge_correction[z] = f

    return rhoana_after_merge_correction

  @staticmethod
  def perform_sim_user_merge_correction(input_image, input_gold, input_rhoana, merge_errors):
      rhoana_after_merge_correction = np.array(input_rhoana)
      
      fixes = []

      for me in merge_errors:
          pred = me[2]
      
          z = me[0]
          label = me[1]
          border = me[3][0][1]
          a,b,c,d,e,f,g,h,i = Legacy.get_merge_error_image(input_image[z], rhoana_after_merge_correction[z], label, border)

          # check VI for this slice
          vi_before = Util.vi(input_gold[z], input_rhoana[z])
          vi_after = Util.vi(input_gold[z], f)
          if (vi_after < vi_before):
            # this is a good fix
            rhoana_after_merge_correction[z] = f
            fixes.append('Good')
          else:
            # skipping this one
            fixes.append('Bad')
            continue            

      return rhoana_after_merge_correction, fixes

  @staticmethod
  def perform_auto_split_correction(cnn, input_image, input_prob, input_rhoana, input_gold, p):

    '''
    Return VI rather than data.
    '''

    vol_vi_s = []
    for z in range(input_image.shape[0]):
      vi_s, merge_pairs, surenesses = Legacy.splits(cnn, input_image[z], input_prob[z], input_rhoana[z], input_gold[z], 
                                                   sureness_threshold=p,
                                                   verbose=False)
      vol_vi_s.append(vi_s[-1])

    return np.mean(vol_vi_s), np.median(vol_vi_s), vol_vi_s


  @staticmethod
  def splits(cnn, image, prob, segmentation, groundtruth=np.zeros((1,1)), sureness_threshold=0., smallest_first=False, oversampling=False, verbose=True, max=10000):
    '''
    '''
    t0 = time.time()
    patches = Patch.patchify(image, prob, segmentation, oversampling=oversampling, max=max)
    if verbose:
      print len(patches), 'generated in', time.time()-t0, 'seconds.'

    t0 = time.time()
    grouped_patches = Patch.group(patches)
    if verbose:
      print 'Grouped into', len(grouped_patches.keys()), 'patches in', time.time()-t0, 'seconds.'


    hist = Util.get_histogram(segmentation.astype(np.float))
    labels = len(hist)

    # create Matrix
    M = np.zeros((labels, labels), dtype=np.float)
    # .. and initialize with -1
    M[:,:] = -1


    for l_n in grouped_patches.keys():

      l = int(l_n.split('-')[0])
      n = int(l_n.split('-')[1])

      # test this patch group for l and n
      prediction = Patch.test_and_unify(grouped_patches[l_n], cnn)

      # fill value into matrix
      M[l,n] = prediction
      M[n,l] = prediction
      

    #
    # NOW the matrix is filled and we can start merging
    #

    # sureness_threshold = 0.
    matrix = np.array(M)
    segmentation_copy = np.array(segmentation)

    if groundtruth.shape[0]>1:
      before_vi = Util.vi(segmentation_copy, groundtruth)

    # we keep track of the following values
    vi_s = []
    ed_s = []
    merge_pairs = []
    surenesses = []

    # now the loop
    t0 = time.time()
    while (matrix.max() >= sureness_threshold):
        
        sureness = matrix.max()
        
        largest_indices = np.where(matrix==sureness)
        #
        #
        # TO TRY: merge smaller ones with smaller ones first
        #



        picked = 0

        if smallest_first:
          smallest = np.Infinity
          smallest_label = -1

          for i,label in enumerate(largest_indices[0]):
            current_size = len(segmentation_copy[segmentation_copy == label])
            if current_size < smallest:
              smallest = current_size
              smallest_label = label
              picked = i


        l,n = largest_indices[0][picked], largest_indices[1][picked]
        

        #
        # TODO also check for alternative n's
        #






        segmentation_copy[segmentation_copy == n] = l
        
        # reset all l,n entries
        matrix[l,:] = -2
        matrix[:,l] = -2
        matrix[n,:] = -2
        matrix[:,n] = -2
        
        if groundtruth.shape[0]>1:
          after_merge_vi = Util.vi(segmentation_copy, groundtruth)
          # after_merge_ed = Util.vi(segmentation_copy, groundtruth)
          vi_s.append(after_merge_vi)

        merge_pairs.append((l,n))
        surenesses.append(sureness)
        
        
        # grab new neighbors of l
        l_neighbors = Util.grab_neighbors(segmentation_copy, l)

        for l_neighbor in l_neighbors:
            # recalculate new neighbors of l
            
            if l_neighbor == 0:
                # ignore neighbor zero
                continue
        
            prediction = Patch.grab_group_test_and_unify(cnn, image, prob, segmentation_copy, l, l_neighbor, oversampling=oversampling)
        
            matrix[l,l_neighbor] = prediction
            matrix[l_neighbor,l] = prediction

    if verbose:
      print 'Merge loop finished in', time.time()-t0, 'seconds.'

    if groundtruth.shape[0]>1:
      min_vi_index = vi_s.index(np.min(vi_s))
      if verbose:
        print 'Before VI:', before_vi
        print 'Smallest VI:', vi_s[min_vi_index]
        print 'Sureness threshold:', surenesses[min_vi_index]


    return vi_s, merge_pairs, surenesses


  @staticmethod
  def VI(gt, seg):
      # total_vi = 0
      slice_vi = []    
      for i in range(gt.shape[0]):
          current_vi = Util.vi(gt[i].astype(np.int64), seg[i].astype(np.int64))
          # total_vi += current_vi
          slice_vi.append(current_vi)
      # total_vi /= 10
      return np.mean(slice_vi), np.median(slice_vi), slice_vi

