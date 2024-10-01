"""
Interface for topological localization.
"""
import os
import torch
import vg_networks.parser_vg as parser
import logging
import pickle

from os.path import join
from datetime import datetime
import torchvision.transforms as T

import vg_networks.util as util
import vg_networks.commons as commons
from vg_networks import network
from topological.colonmapper import bayesian_localization_images
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

######################################### MODEL #########################################

# Extract configuration from the name of the folder. It is the parent folder of the model
model_conf = os.path.basename(os.path.dirname(args.resume))
new_args, short_sim, radenovic = util.get_configuration(model_conf)
args.__dict__.update(new_args)
print(args.resize)
print(f'Aggregator is {args.aggregation}')

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
print('>> {}: Load datasets...'.format(args.datasets))

save_folder = EVAL_PATH / str(model_conf)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for dataset in args.datasets:
    # Load images
    if dataset.startswith('027'):
        print('>> {}: Bayesian localization for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_027/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][4981:]
        graph_name = 'entry_027.pkl'
        initial_node = -1
    elif dataset.startswith('035'):
        print('>> {}: Bayesian localization for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][7230:]
        graph_name = 'entry_035.pkl'
        initial_node = -1
    elif dataset.startswith('entry_cross'):
        print('>> {}: Bayesian localization for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][:7230]
        graph_name = 'withdrawal_027.pkl'
        initial_node = 114
    elif dataset.startswith('cross'):
        print('>> {}: Bayesian localization for sequence '.format(dataset))
        images_folder = DATA_PATH / 'endomapper/Seq_035/images'
        images = os.listdir(images_folder)
        images.sort()
        images = [os.path.join(images_folder, image) for image in images][7230:]
        graph_name = 'withdrawal_027.pkl'
        initial_node = 0
    elif dataset.startswith('C3VD'):
        print('>> {}: Bayesian localization for dataset '.format(dataset))
        initial_node = 0
        c3vd_path = DATA_PATH / 'C3VD' 
        sequence_list = ['seq2', 'seq3', 'seq4']
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
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'resize': [675, 506],
        }


    if dataset.startswith('C3VD'):
        # Always use canonical maps
        graph_pkl = DATA_PATH / 'C3VD/canonical_maps/seq1_graph.pkl'
        print('>> {}: Use canonical map... {}'.format(dataset, graph_pkl))

        with open(graph_pkl, 'rb') as f:
            graph_dict = pickle.load(f)

        # Load reject node
        if args.reject_outliers:
            reject_path = DATA_PATH / 'endomapper/reject_node'
            print('>> {}: Loadin reject node files... {}'.format(dataset, reject_path))
            reject = os.listdir(reject_path)
            reject.sort()
            reject = [os.path.join(reject_path, image) for image in reject]
        else:
            reject = None

        for seq, images in sequence_images.items():
            conf = {'radius_bayesian': 2,
                    'likelihood': args.likelihood_method,
                    'reject_outliers': args.reject_outliers,
                    'reject_strategy': args.reject_strategy,
                    'threshold_probability': args.threshold_probability,
                    'threshold_likelihood': args.threshold_likelihood,
                    'experiment_folder': join(save_folder, args.experiment_name, dataset),
                    'initial_node': initial_node,
                    'save_path': join(save_folder, dataset, seq),
                    'short_sim': short_sim,
                    'multiple_to_localize': 5,
                    'verification': verifier_config} 
        
            if not os.path.exists(conf['save_path']):
                print('>> {}: Create save path...'.format(conf['save_path']))
                os.makedirs(conf['save_path'])

            start_time = datetime.now()
            print('>> {}: Localizing...'.format(dataset))

            graph, results = bayesian_localization_images(conf, graph_dict, model, images, reject, args.resize, args.features_dim, transform=base_transform, vg=True, plot=args.plot, bayesian=args.bayesian, reload_graph=args.reload_graph)

            logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
            graph.save_results_text_c3vd(results, seq, save_folder, experiment_name=args.experiment_name)

            print('>> {}: Probability was saved.'.format(dataset))

    else:
        ## ENDOMAPPER DATASETS ##

        # Always use canonical maps
        graph_pkl = DATA_PATH / 'endomapper/canonical_maps'/ graph_name
        print('>> {}: Use canonical maps... {}'.format(dataset, graph_pkl))
        
        with open(graph_pkl, 'rb') as f:
            graph_dict = pickle.load(f)

        # Load reject node
        if args.reject_outliers:
            reject_path = DATA_PATH / 'endomapper' / 'reject_node'
            print('>> {}: Loadin reject node files... {}'.format(dataset, reject_path))
            reject = os.listdir(reject_path)
            reject.sort()
            reject = [os.path.join(reject_path, image) for image in reject]
        else:
            reject = None

        print('>> {}:{} Run Bayesian localization...'.format(args.experiment_name, dataset))

        conf = {'radius_bayesian': 2,
                'likelihood': args.likelihood_method,
                'reject_outliers': args.reject_outliers,
                'reject_strategy': args.reject_strategy,
                'threshold_probability': args.threshold_probability,
                'threshold_likelihood': args.threshold_likelihood,
                'experiment_folder': join(save_folder, args.experiment_name, dataset),
                'initial_node': initial_node,
                'save_path': join(save_folder, dataset),
                'short_sim': short_sim,
                'multiple_to_localize': 5,
                'verification': verifier_config}  
        
        if not os.path.exists(conf['save_path']):
            print('>> {}: Create save path...'.format(conf['save_path']))
            os.makedirs(conf['save_path'])

        start_time = datetime.now()
        print('>> {}: Localizing...'.format(dataset))

        graph, results = bayesian_localization_images(conf, graph_dict, model, images, reject, args.resize, args.features_dim, transform=base_transform, vg=True, plot=args.plot, bayesian=args.bayesian, reload_graph=args.reload_graph)

        logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
        graph.save_results_text_endomapper(results, dataset, save_folder, experiment_name=args.experiment_name)

        print('>> {}: Localizations were saved.'.format(dataset))