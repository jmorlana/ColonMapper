"""
Visualize the images used for evaluation in Seq_027 and Seq_035 in the Endomapper dataset (1 out of 5).
"""
import os
import vg_networks.parser_vg as parser
import logging

from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
import torchvision.transforms as T

import vg_networks.commons as commons
import cv2
import tqdm
from settings import DATA_PATH, EVAL_PATH

logging.getLogger('PIL').setLevel(logging.WARNING)

def take_int(elem):
    # Get filename from path
    elem = os.path.basename(elem)
    if 'color' in elem:
        return int(elem.replace('_color','').replace('.png',''))
    return int(elem.replace('.png',''))


######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### DATASET #########################################

# evaluate on test datasets
#datasets = args.datasets.split(',')
print('>> {}: Load datasets...'.format(args.datasets))


for dataset in args.datasets:
    # Load images
    if dataset.startswith('027'):
        print('>> {}: Bayesian localization for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_027/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][4981:]
        save_path = EVAL_PATH / 'eval_027'
    elif dataset.startswith('035'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][7230:]
        save_path = EVAL_PATH / 'eval_035'
    elif dataset.startswith('cross'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][7230:]
        save_path = EVAL_PATH / 'eval_cross'
    elif dataset.startswith('entry_cross'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][:7230]
        save_path = EVAL_PATH / 'eval_entry_cross'
    
        
    if not os.path.exists(save_path):
        print('>> {}: Create save path...'.format(save_path))
        os.makedirs(save_path)

    # Get 1 image out of 5
    images = images[::5]

    # Read and resize images to 640x480 and save them into save_path
    print('>> {}: Read and resize images...'.format(dataset))
    # Use tqdm to show progress bar and get number of image
    n = 0
    for image in tqdm.tqdm(images):
        img = cv2.imread(image)
        img = cv2.resize(img, (640,480))
        filename = str(n).zfill(4) + '_' + os.path.basename(image) 
        cv2.imwrite(os.path.join(save_path, filename), img)
        n += 1
