# Adapted by Sergio A Velastin 2023
# In this version we define a function for segmenting one object (to start making things modular)
# we can then process a number of images
# and then process all the GT bboxes of the given class (see "classes")

# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Object masks from prompts with SAM
# The Segment Anything Model (SAM) predicts object masks given prompts that indicate the desired object. The model first converts the image into an image embedding that allows high quality masks to be efficiently produced from a prompt.

# The `SamPredictor` class provides an easy interface to the model for prompting the model. It allows the user to first set an image using the `set_image` method, which calculates the necessary image embeddings. Then, prompts can be provided via the `predict` method to efficiently predict masks from those prompts. The model can take as input both point and box prompts, as well as masks from the previous iteration of prediction.

# Support packages
import os
import xmltodict
import glob
from pause import *
import shutil
import json
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
# https://github.com/jsbroks/imantics
from imantics import Polygons, Mask     # to work out polygons from masks
from segment_anything import sam_model_registry, SamPredictor

CHECKPOINT_DEFAULT='sam_vit_h_4b8939.pth'
LOGFILE= 'log_segment_GT2YOLO.log'
IMGDIRNAME = 'images'
LABDIRNAME = 'labels'

#******************************* Support Functions ***********************************
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100):   # was 375
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def stop_here(msg=''):
    if msg != '':
        print(msg)
    exit(1)

#************************************   Display an image and bbox and marker points
def display_image (image, name, all_ids, all_centroids, all_labels, input_box, classes):
    # All the image done, now we can display the image
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    print(image.shape[0:2])
    height, width = image.shape[0:2]
    for idx, id in enumerate(all_ids):  #TODO: could use zip instead of list indeces
        if not id in classes:
            continue
        centroid_points=all_centroids[idx]
        centroid_labels=all_labels[idx]
        test_box=input_box[idx]
        show_points(centroid_points, centroid_labels, plt.gca())
        show_box (test_box, plt.gca())
    plt.title(f'{name}, bbox and points w,h: {width}, {height}')
    plt.axis('on')
    plt.show()

#******************************  Function to obtain masks for a given object (bbox) *****************
# centre_points: indicates if we use 5 points (centroid+around centroid)
# centre_factor: when using 5 points, is percentage of original width/height used for the extra 4 points 
# use_bbox: Use the input_box as a marker too
def get_mask (predictor, input_box, centre_points=True, centre_factor=0.4, use_bbox=True, add_points=[]):
    # Work out w, h, centroid
    width, height = input_box[2]-input_box[0], input_box[3]-input_box[1]
    xc, yc = int((input_box[2]+input_box[0])/2), int((input_box[3]+input_box[1])/2)
    # We will choose a positive marker in the centre of the object
    centroid_points = [[xc,yc]]
    centroid_labels =[1]  # a positive point

    # Check if we are adding extra points
    if centre_points:
        logging.debug(f'before {centroid_points}')
        small_w=int(width*centre_factor/2)     # a smaller bbox (/2 for convenience, see below)
        small_h=int(height*centre_factor/2)
        # In theory, no risk of going outside image as this is a smaller box
        centroid_points.append([xc-small_w, yc-small_h])     # top left
        centroid_labels.append(1)                            # we should end with 5 points, done for clarity
        centroid_points.append([xc+small_w, yc-small_h])    # top right
        centroid_labels.append(1)
        centroid_points.append([xc-small_w, yc+small_h])     # bottom left
        centroid_labels.append(1)
        centroid_points.append([xc+small_w, yc+small_h])     # bottom right
        centroid_labels.append(1)
    # See if there are additional points (normally from heuristics)
    for elem in add_points:
        centroid_points.append (elem)
        centroid_labels.append(1)
        
    centroid_points=np.array(centroid_points)
    centroid_labels=np.array(centroid_labels)
    logging.debug(f'after np {centroid_points} {centroid_labels}')

    if use_bbox:    #TODO: maybe setting box:None might cover the case when not using bbox
        masks, scores, logits = predictor.predict(
            point_coords=centroid_points,
            point_labels=centroid_labels,
            multimask_output=True,
            box=input_box)
    else:
        masks, scores, logits = predictor.predict(
            point_coords=centroid_points,
            point_labels=centroid_labels,
            multimask_output=True)
    return masks, scores, logits, centroid_points, centroid_labels

# Changes an array of coordinates assumed to be in the order xy, to relative coordinates
# coords is the segmentation list returned by get_segmentation()
# NOTE: it would have been neater to use 'points' returned by get_segmentation, but such is life!
def relative_xy (coords, width, height):
    assert (len(coords)%2) == 0, f'length of coords list/array {len(coords)} should be even'
    output=''
    for idx,_ in enumerate(coords):
        if idx%2 != 0:
            continue    # we have done it for odd elements
        output += f'{coords[idx]/width:4f} {coords[idx+1]/height:4f} '
    #print(coords, width, height, output)
    return output   # for convenience, to output directly to a file

# Returns True if a point [x,y] is inside a box [xmin, ymin, xmax, ymax]
def point_inside (point, box):
    x,y = point         # not efficient but clear
    xmin, ymin, xmax, ymax = box
    if (x>xmax) or (x<xmin):
        return False
    if (y>ymax) or (y<ymin):
        return False
    return True
    
# Here we would generate the YOLO GT corresponding to ONE of the input bboxes
# test_box i the bounding box (ideally it just encloses all segmentation, but not always so)
# segmentation: is the polygons.segmentation
# NOTE: it would have been neater to use 'points' returned by get_segmentation, but such is life!
def process_segmentation (id, width, height, test_box, segmentation, out_file=None):
    logging.debug(f'Process Segmentation id {id} width {width} height {height} testbox {test_box}\n segmentation {segmentation}')
    assert (len(segmentation)%2) == 0, f'length of segmentation {len(segmentation)} should be even'
    if out_file != None:
        out_file.write(f'{id} ')
    # we output test_box (relative to weight/height
    # We do NOT output the bbox, just the polygons
    #output= relative_xy (test_box, width, height)
    #if out_file != None:
    #    out_file.write(f'{output}')
    #logging.debug(f'Relative bbox: {output}')
    output = relative_xy (segmentation, width, height)
    if out_file != None:
        out_file.write(f'{output}\n')
    logging.debug(f'Relative segmentation: {output}')

# produces a set of segmentation (polygon) points from a mask, by default it uses imantics package
# as I have not time to find a way of converting the mask to an OpenCV image without writing to a file!
# bbox: object's GT bbox
# lbox: if True, then the polygon points are clipped to fall withing the bbox, else bbox is ignored
def get_segmentation(mask, bbox, lbox, useOpenCV=False):
    if useOpenCV:
        TEMP_FILE='test.png'
        new_mask = np.asarray(mask, dtype=np.int8)
        # This is the OpenCV code to process the mask
        # TODO: find out how to avoid writing to a file
        cv2.imwrite(TEMP_FILE, new_mask)
        logging.debug('temp file has been written')
        # Now read back the image as a grey image
        new_image = cv2.imread(TEMP_FILE, cv2.IMREAD_GRAYSCALE)
        #print('temp file has been read')
        contours, heirarchy = cv2.findContours(new_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # It might be possible that we got more than one contour because mask is broken up
        logging.debug(f'heirarchy {heirarchy}')
        logging.debug(f'len of contours is: {len(contours)}')
        if len(contours)==0:
            return [],[]
        contour_elems = []
        for idx, elem in enumerate(contours):
            contour_elems.append(len(elem))
            logging.debug(f'{idx}:{len(elem)}')
        max_elem = np.argmax(contour_elems)
        logging.debug(f'we should go for {max_elem}')
        # We now build the polygons:
        segmentation=[]
        points=[]
        for elem in contours[max_elem]:    # [0] as only one set of contours because one object
            segmentation.append(elem[0][0]) # elem is a one-element list of a list of x y
            segmentation.append(elem[0][1])
            points.append([elem[0][0], elem[0][1]])
        segmentation=np.asarray(segmentation)
        points=np.asarray(points)
    else:
        # Here we try with imantics (they might use OpenCV anyway!)
        polygons = Mask(mask).polygons()
        logging.debug(f'{len(polygons.segmentation)} {len(polygons.points)}')
        if len(polygons.points)==0:
            return [],[]
        contour_elems = []
        for idx, elem in enumerate(polygons.points):
            contour_elems.append(len(elem))
            logging.debug(f'{idx}:{len(elem)}')
        max_elem = np.argmax(contour_elems)
        logging.debug(f'we should go for {max_elem}')
        segmentation=polygons.segmentation[max_elem]
        points=polygons.points[max_elem]
    logging.debug(f'segmentation {segmentation}')
    logging.debug(f'points {points}')
    if not lbox:
        return segmentation, points
    # We need to remove any points that are outside the bbox
    #print(f'Before pruning {len(segmentation)} segs, {len(points)} points')
    new_segmentation=[]
    new_points=[]
    # order in bbox: [xmin,ymin,xmax,ymax]
    xmin,ymin,xmax,ymax=bbox
    assert len(segmentation)==2*len(points), f'Expected that len(segmentation): {len(segmentation)} is 2*len(points): {len(points)}'
    for idx, point in enumerate(points):
        sidx = 2*idx    # index for "flat" segmentation list
        x,y = point
        if (xmin<=x) and (x<= xmax) and (ymin<=y) and (y<=ymax):    # all is well
            #print(f'{bbox}, {point}: All ok')
            continue
        if (x<xmin):
            x=xmin
        if (x>xmax):
            x=xmax
        if (y<ymin):
            y=ymin
        if (y>ymax):
            y=ymax
        points[idx]=[x,y]
        segmentation[sidx]= x
        segmentation[sidx+1] = y
        #print(f'{bbox}, {points[idx]}: Clipped')
    #print(f'after clipping {len(segmentation)} segs, {len(points)} points')
    return segmentation, points

# ************************************   Parser
def parse_args(args):
    """ Parse the arguments., '-' used for required and '--' for optional arguments
    """
    parser = argparse.ArgumentParser(description='Runs SAM segmentation for a set of image files/annotations')
    parser.add_argument('-i', help='Path to the input image root', required=True, type=str)
    parser.add_argument('-x', help='Path to the XML annotations', required=True, type=str)
    parser.add_argument('-o', help='Path to the output directory (if it exists, it is deleted!)', required=True,
                        type=str)

    parser.add_argument('--ckp', help=f'Path for check point file (default: {CHECKPOINT_DEFAULT})', type=str)
    parser.add_argument('--cfact', help='Relative size of box to locate point markers (default=0.6)', type=float,
                        default=0.6)
    parser.add_argument('--ncp', help='Do NOT use additional point markers', action='store_true')
    parser.add_argument('--nbox', help='Do NOT use GT bbox as a marker', action='store_true')
    parser.add_argument('--nlim', help="Do NOT limit output polygon to GT bbox", action='store_true')
    parser.add_argument('--classes', help='Path to json file with classes to use (default=["object"])', type=str,
                        default='object')
    parser.add_argument('--ln', help='if output images are links (else they are copied)', action='store_true')
    parser.add_argument('--pause', help='If pausing at various points', action='store_true')
    parser.add_argument('--verbose', help='to use logging.debug for additional output', action='store_true')
    parser.add_argument('--show', help='If showing visual results (quite slow!)', action='store_true')
    parser.add_argument('--rc', help='Use random colours for mask display', action='store_true')
    parser.add_argument('--heur1', help='Apply a heuristic rule', action='store_true')
    parser.add_argument('--heur2', help='Apply another heuristic rule', action='store_true')
    return parser.parse_args(args)

# ********************************** Heuristics: ad-hoc rules  *******************
# These do nothing now. Retained as examples
def heuristics1(HEUR, class_id, test_box, objects, centre_factor):

    add_points = []
    #print(f'class id {class_id}')
    if HEUR and (class_id=='whatever'):
        # We add points along the mid height
        #xmin, ymin, xmax, ymax = test_box
        width, height = int((xmax-xmin)/2), int((ymax-ymin)/2)
        xc, yc = int((xmax+xmin)/2), int((ymax+ymin)/2)
        spacing = 20
        delta = width/spacing*2
        for idx in range(spacing):
            add_points.append([xmin+delta*idx, yc])
    return add_points

# ***********************************   Special case, working on mask  ************************
# mask: best mask for the given (shelf) object
def heuristics2(HEUR2, idx, mask, all_ids, all_masks, all_scores, all_boxes):
    new_mask= mask          # the default is that we return the same mask
    class_id = all_ids[idx]

    #print(f'class id {class_id}')
    if HEUR2 and (class_id=='whatever'):
        test_box=all_boxes[idx]
        xmin, ymin, xmax, ymax = test_box
        width, height = int((xmax-xmin)/2), int((ymax-ymin)/2)
        xc, yc = int((xmax+xmin)/2), int((ymax+ymin)/2)
        print (f'{class_id}: {xmin} {ymin} {xmax} {ymax}')
        # and we do nothing else! (we could scan objects making decisions to change the mask etc.)
    return new_mask


#**********************************   Main program  ****************
# ## Example image
def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    SetPause(args.pause)
    SHOW=args.show
    VERBOSE = args.verbose
    if (VERBOSE):
        logging.basicConfig(format="[%(asctime)s: %(name)s] %(levelname)s\t%(message)s",
                            handlers=[
                                logging.FileHandler(LOGFILE),
                                logging.StreamHandler()
                            ],
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format="[%(asctime)s: %(name)s] %(levelname)s\t%(message)s",
                            handlers=[
                                logging.FileHandler(LOGFILE),
                                logging.StreamHandler()
                            ],
                            level=logging.INFO)
    logging.info ('Process Started')
    logging.info(f'{args}')
    
    IMAGE_ROOT = os.path.normpath(args.i)
    if not os.path.isdir(IMAGE_ROOT):
        logging.critical('*** Oops!, ImageRoot does not exist '+ IMAGE_ROOT)
        exit(1)
    # Ditto for XMLroot
    XML_ROOT=os.path.normpath(args.x)
    if not os.path.isdir(XML_ROOT):
        logging.critical('*** Oops!, XMLroot does not exist '+ XML_ROOT)
        exit(2)

    CENTRE_FACTOR= args.cfact
    CENTRE_POINTS= not args.ncp
    USE_BBOX = not args.nbox
    LIMIT_BBOX = not args.nlim
    HEUR1 = args.heur1
    HEUR2 = args.heur2
    logging.debug(f'CENTRE_FACTOR: {CENTRE_FACTOR}, CENTRE_POINTS: {CENTRE_POINTS}, USE_BBOX: {USE_BBOX}, LIMIT_BBOX: {LIMIT_BBOX}')
    
    classes = []   # This is the default:
    if args.classes != '':
        jsonclasses = os.path.abspath(args.classes)
        pause(jsonclasses+' class file')
        if not os.path.isfile (jsonclasses):
            logging.critical(f'Oops!, json file does not exist {jsonclasses}')
            exit(2)
        with open(jsonclasses, 'r') as filehdl:
            classes = json.load(filehdl)
            if classes == []:
                logging.critical('Oops!, empty set of classes!')
                exit(2)
    if classes ==[]:
        classes =['object']
        logging.info(f'Classes set to: {classes}')
    pause(f'Classes: {classes}')
    logging.debug (f'Classes: {classes}')

    OutputRoot = os.path.normpath(args.o)
    # It is better to make sure we are creating a clean output directory
    if os.path.isdir(OutputRoot):
        hardpause(f'Warning! directory {OutputRoot} will be deleted (you can abort with Ctrl-C now)')
        hardpause(f'Last chance to abort this script!')
        shutil.rmtree (OutputRoot)
    os.makedirs(OutputRoot)
    OutputRootImages = os.path.join(OutputRoot,IMGDIRNAME)
    OutputRootLabels = os.path.join(OutputRoot,LABDIRNAME)
    os.makedirs(OutputRootImages)    # and the sub-dirs
    os.makedirs(OutputRootLabels)
    logging.info(f'Directories {OutputRootImages} and {OutputRootLabels} created')

    # First, load the SAM model and predictor. Change the path below to point to the SAM checkpoint. Running on CUDA and using the default model are recommended for best results.
    sam_checkpoint = CHECKPOINT_DEFAULT  # "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    logging.info('loading checkpoint...')
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    logging.info('Setting up predictor')
    predictor = SamPredictor(sam)
    # stop_here('just after predictor')

    Image_files=glob.glob(os.path.join(IMAGE_ROOT, '*'))
    n_files = len(Image_files)
    n_xmlfiles=0
    logging.info(f'Images: {Image_files}')

    for idx, Image_file in enumerate(Image_files):
        basename, extension = os.path.splitext (os.path.basename(Image_file))  # separate extension e.g. ".jpg"
        image = cv2.imread(Image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0:2]
        logging.info(f'{idx+1}/{n_files} Image: {basename}')

        Xml_file = os.path.join(XML_ROOT, basename)+'.xml'
        if not os.path.isfile(Xml_file):
            logging.error('*** buahhhh '+Xml_file+' does not exist')
            continue
        n_xmlfiles += 1
        predictor.set_image(image)
        logging.debug('predictor set up')
        # Here we decode the XML annotations
        with open(Xml_file, "r", encoding="utf8") as datafile:
            annotations = xmltodict.parse(datafile.read())
        if 'object' in annotations['annotation']:
            objects = annotations['annotation']['object']
            if type(objects) is not list:                  # happens when we only have one object
                objects=[objects]
        else:
            logging.warning (f"*** no object in xml file {Xml_file}")
            objects = []    # no objects   NOTE: be careful if following code can cope with this

        # We now build a list of boxes a-la-SAM
        # We predict bbox by bbox and collect results
        # So we do each object (class=id) SEPARATELY and we get full image size masks of each one
        # That could be quite wasteful of memory and time!, but it might help with finding contours (polygons)
        # TODO: it might be more efficient to batch these as per the SAM's example code
        # all_ids means all the objects, all_ids[i] is ith object id (class) and all_masks[i] is masks for ith object, etc.
        all_boxes, all_ids, all_masks, all_scores, all_centroids, all_labels =[],[],[],[],[],[]
        for idx, obj in enumerate(objects):
            id = obj['name']    # class id e.g. 'object'
            if HEUR2 or (id in classes):   # if we are processing this type of object, NOTE: HEUR2 forces all objects
                xmin = int(float(obj['bndbox']['xmin']))    #SAV 29.Apr.2022 prevents error when string contains a float
                ymin = int(float(obj['bndbox']['ymin']))
                xmax = int(float(obj['bndbox']['xmax']))
                ymax = int(float(obj['bndbox']['ymax']))
                #print (f'{xmin} {ymin} {xmax} {ymax}')
                test_box = np.array([xmin,ymin,xmax,ymax])
                logging.debug(f'Processing obj {idx}:{id}, {test_box} ')
                all_boxes.append(test_box)
                add_points = heuristics1(HEUR1, id, test_box, objects, CENTRE_FACTOR)
                logging.debug(f'add_points: {add_points}')

                # NOTE: we have to wait for segmentation before we can display original image and bboxes etc!
                # masks returns 3 masks with 3 scores which we later use to extract the best of 3 masks
                masks, scores, logits, centroid_points, centroid_labels = get_mask (predictor, test_box,
                                                                                    centre_points=CENTRE_POINTS,
                                                                                    centre_factor=CENTRE_FACTOR,
                                                                                    use_bbox=USE_BBOX,
                                                                                    add_points= add_points)
                logging.debug(f'{idx}, scores: {scores}')
                all_ids.append(id)
                all_masks.append(masks)
                all_scores.append(scores)
                all_centroids.append(centroid_points)
                all_labels.append(centroid_labels)

        # All the image done, now we can display the image
        if SHOW:
            display_image(image, basename, all_ids, all_centroids, all_labels, all_boxes, classes)
        
        # First we can copy the image and create a corresponding image file
        src = os.path.abspath(Image_file)   # better for linking
        if args.ln:
            # We will create a symbolic link
            dst = os.path.join(OutputRootImages, f'{basename}{extension}')
            os.symlink(src, dst)
        else:
            dst = OutputRootImages
            shutil.copy2(src, dst)
        logging.info(f'{src} --> {dst}')
        # Now we create a label file
        dst=os.path.join(OutputRootLabels, basename+'.txt')
        logging.info (f'Label file: {dst}')
        out_file = open(dst, 'w')

        # And now we can compute and display all the segmentation results
        if SHOW:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
        for idx, id in enumerate(all_ids):  #TODO: could use zip instead of list indeces
            if not (id in classes):
                continue                # we do not care about objects not in classes
            masks = all_masks[idx]
            scores= all_scores[idx]
            max_score = np.argmax(scores)   # index to highest mask
            logging.debug(f'scores {scores}, max {max_score}')
            mask = masks[max_score]         # the corresponding mask

            # Here we apply a heuristic rule based on the mask
            new_mask = heuristics2(HEUR2, idx, mask, all_ids, all_masks, all_scores, all_boxes)   # in case we need to apply a rule

            if SHOW:
                show_mask(new_mask, plt.gca(), random_color=args.rc)      # display it
            centroid_points=all_centroids[idx]
            centroid_labels=all_labels[idx]
            test_box=all_boxes[idx]
            logging.debug(f'test box: {test_box}')
            logging.debug(f'mask: {mask}')
            segmentation, points = get_segmentation(new_mask, bbox=test_box, lbox= LIMIT_BBOX, useOpenCV=False)
            #print(f'After get_segmentations: {len(segmentation)} segs, {len(points)} points')
            if len(points)==0:  # no contours were found, so we ignore this object
                logging.warning(f'WARNING: no contours, {idx} {id} {basename}')
                continue
            process_segmentation(classes.index(id), width, height, test_box, segmentation, out_file=out_file) # note [0]
            if SHOW:
                show_box (test_box, plt.gca())              # the bounding box
                labels = np.zeros(len(points), dtype=int)
                logging.debug(f'labels: {labels}\n{len(labels)}')
                show_points(points, labels, plt.gca(), marker_size=2)
                #print('show points worked')
                #show_points(centroid_points, centroid_labels, plt.gca())    # dont show, too cluttered
        if SHOW:
            plt.title('Segmentation using bbox and centre points')
            plt.axis('on')
            plt.show()
        out_file.close()
    logging.info(f'Job completed with {n_files} images and {n_xmlfiles} annotations')
    if n_files != n_xmlfiles:
        logging.error('mhhh, some XML files not found?')

if __name__ == '__main__':
     main()


