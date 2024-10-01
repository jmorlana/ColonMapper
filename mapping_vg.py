"""
Interface for topological mapping.
"""
import os
import torch
import vg_networks.parser_vg as parser
import logging
from os.path import join
from datetime import datetime
import torchvision.transforms as T

import vg_networks.util as util
import vg_networks.commons as commons
from vg_networks import network
from topological.colonmapper import topological_mapping
from settings import DATA_PATH, EVAL_PATH

logging.getLogger('PIL').setLevel(logging.WARNING)

def take_int(elem):
    # Get filename from path
    elem = os.path.basename(elem)
    return int(elem.replace('.png',''))

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################

# Extract configuration from the name of the folder. It is the parent folder of the model
model_conf = os.path.basename(os.path.dirname(args.resume))
new_args, short_sim, radenovic = util.get_configuration(model_conf)
args.__dict__.update(new_args)
print(args.resize)

model = network.GeoLocalizationNet(args)
model = model.to(args.device)

if args.aggregation in ["netvlad", "crn"]:
    args.features_dim *= args.netvlad_clusters

# Load the model from path
if radenovic:
    logging.info(f"Loading Radenovic model from {args.resume}")
    state = torch.load(args.resume)
    state_dict = state["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)
else:
    logging.info(f"Resuming newest model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)
model.eval()
model.cuda()

# Transformations
base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

######################################### DATASET #########################################

# evaluate on test datasets
save_folder = EVAL_PATH / str(model_conf)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for dataset in args.datasets:
    # Load images
    if dataset.startswith('027'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_027/images'
        images = os.listdir(images_folder)
        images.sort()
        # create map with entry phase of Seq_027, to localize later withdrawal of Seq_027
        images = [os.path.join(images_folder, image) for image in images][:4981] 
        graph_name = 'entry_027.pkl'
    elif dataset.startswith('035'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        # create map with entry phase of Seq_035, to localize later withdrawal of Seq_035
        images = [os.path.join(images_folder, image) for image in images][:7230] 
        graph_name = 'entry_035.pkl'
    elif dataset.startswith('cross'):
        print('>> {}: Topological mapping for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_027/images'
        images = os.listdir(images_folder)
        images.sort()
        # create map with withdrawal phase of Seq_027, to localize later withdrawal of Seq_035
        images = [os.path.join(images_folder, image) for image in images][4981:]
        graph_name = 'withdrawal_027.pkl'
    elif dataset.startswith('C3VD'):
        print('>> {}: Topological mapping for dataset '.format(dataset))
        c3vd_path = DATA_PATH / 'C3VD' 
        sequence_list = ['seq1', 'seq2', 'seq3', 'seq4']
        sequence_images = {}
        for sequence in sequence_list:
            images_folder = os.path.join(c3vd_path, sequence)
            images = os.listdir(images_folder)

            # Remove files that are not images
            images = [image for image in images if image.endswith('.png')]

            # Sort images by number
            images = sorted(images, key = take_int)
            images = [os.path.join(images_folder, image) for image in images if image.endswith('.png')]
            sequence_images[sequence] = images

    # Configs for local matching
    verifier_config = {
            'use_nn_matcher': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resize': [675, 506],
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.001,
                'max_keypoints': 4096
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            },
            'nn': {
                'distance_thresh': 0.7,
                'mutual_check': False,
            },
        }
    
    if dataset.startswith('C3VD'):
        print('>> {}: Run topological mapping...'.format(dataset))
        for seq, images in sequence_images.items():
            print('>> {}: Sequence for mapping is {}...'.format(dataset, seq))

            graph_name = seq + '_graph.pkl'

            conf = {'radius_bayesian': 2,
                'multi_descriptor_node': False,
                'max_skipped': 10,
                'matching_node': 'half',
                'save_path': join(save_folder, dataset, graph_name),
                'short_sim': short_sim,
                'verification': verifier_config}
            
            if not os.path.exists(join(save_folder, dataset)):
                os.makedirs(join(save_folder, dataset))

            start_time = datetime.now()
            graph = topological_mapping(conf, model, images, args.resize, args.features_dim, transform=base_transform, vg=True, plot=args.plot)
            logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

            graph.save_graph()
            print('>> {}: Graph was saved.'.format(seq))

    else:
        print('>> {}: Run topological mapping...'.format(dataset))

        conf = {'radius_bayesian': 2,
            'multi_descriptor_node': False,
            'max_skipped': 7,
            'matching_node': 'last',
            'save_path': join(save_folder, graph_name),
            'short_sim': short_sim,
            'verification': verifier_config}

        start_time = datetime.now()
        graph = topological_mapping(conf, model, images, args.resize, args.features_dim, transform=base_transform, vg=True, plot=args.plot)
        logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

        graph.save_graph()
        print('>> {}: Graph was saved.'.format(dataset))