from collections import OrderedDict
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import os.path
import pickle

from os.path import join, exists

import cv2
import kornia as K
import kornia.feature as KF
from settings import root, DATA_PATH

from .node import MultiNode

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class TopologicalMapLocalization:
    def __init__(self, conf, likelihood_estimator, vg=False, reject=None, width=250, height=200):
        self.radius_bayesian = conf['radius_bayesian']
        self.save_path = conf['save_path']
        self.last_image_inserted = 0
        self.threshold_probability = conf['threshold_probability']
        self.reject_strategy = conf['reject_strategy']

        self.nx_graph = nx.Graph()
        self.nodes = OrderedDict()
        self.n_connections = OrderedDict()
        self.current_position = -1
        self.last_inserted = None

        self.first_plot = True
        self.read_image = True
        self.most_similar = -1
        self.p_kt_pred = []
        self.p_kt = []
        self.scores = []
        self.scores_sg = []
        self.aggregated_p_kt = []

        self.device = conf['verification']['device']
        self.resize = conf['verification']['resize']
        self.sg_matches = None

        self.mask = self.load_mask()
        self.saved_frames = 0

        self.saved_likelihood = []
        self.saved_probabilities = []
        self.single_prob = []
        self.found_loops = []
        self.GeM_times = []
        self.LoFTR_times = []


        # verification
        if conf['verification'] is not None:
            self.matcher = KF.LoFTR(pretrained='outdoor').eval().to(self.device)
            self.verification = True
            self.do_viz = False
        else:
            self.matcher = None
            self.verification = False
            self.do_viz = False

        # likelihood
        self.likelihood_estimator = likelihood_estimator

        # reject (no loop-closure) probabilities
        self.reject_outliers = conf['reject_outliers']
        if reject is not None:
            self.reject = reject
        else:
            self.reject = None
        self.reject_descriptors = None
        self.reject_prediction = 0.0
        self.reject_probability = 0.0
        self.reject_likelihood = 0.0
        self.reject_scores = None
        self.is_reject = False

        self.image_width = width
        self.image_height = height

    def load_mask(self):
        img = cv2.imread(str(root / 'assets/mask.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)  # change the kernel size as needed
        mask = cv2.erode(mask, kernel, iterations=15)
        mask_tensor = K.image_to_tensor(mask, False).float()
        return mask_tensor.squeeze(0).to(self.device)

    # --- GRAPH MODIFICATION ---
    def initialize_graph(self, graph_dict, initial_node = None):
        # Load nodes and images
        ids = graph_dict['node_ids']
        w_images_paths = graph_dict['images']
        
        # Change path
        images_paths = []
        for node in w_images_paths:
            u_node = []
            for image in node:
                image = os.path.join(*image.split(os.sep)[4:])
                image = str(DATA_PATH / image)
                u_node.append(image)
            images_paths.append(u_node)
        n_frames = graph_dict['n_frames']

        # Account for reject
        if self.reject is not None:
            shared_probability = 1.0 / (len(ids) + 1)
            self.reject_probability = shared_probability
        else:
            shared_probability = 1.0 / len(ids) 

        # Add nodes to graph
        for node_id in ids:
            proto_node = MultiNode(graph_dict['descriptors'][node_id], images_paths[node_id], n_frames[node_id], node_id, probability = shared_probability)
            self.add_protonode(proto_node, self.current_position)

        # Assume we start at initial_node
        if initial_node is not None:
            print('>> Starting at node {}'.format(ids[int(initial_node)]))
            self.nodes[ids[int(initial_node)]].probability = 0.9

        # Normalize probabilities
        self.normalize_probabilities_after_initialization()
        self.current_position = 0


    def add_protonode(self, proto_node, node_to_connect = -1):
        node = proto_node

        print('Nodes connected', node.id, '->', node_to_connect)
        if self.read_image:
            node.image = cv2.imread(node.image_paths[0])
            node.image = cv2.resize(node.image, (self.image_width, self.image_height))
            node_text = str(node.id) + ' - ' + str(len(node.image_paths))
            cv2.putText(node.image, node_text, (20, 50), font, fontScale, (10, 255, 10), lineType)
            #cv2.putText(node.image, str(node.n_frame), (20, 80), font, 1, (10, 255, 10), 1)

        if (node_to_connect != -1):
            self.nx_graph.add_node(node.id)
            self.nx_graph.add_edge(self.nodes[node_to_connect].id, node.id)

            self.nodes[node.id] = node
        else:
            # First node doesn't have initial edges
            self.nx_graph.add_node(node.id)
            self.nodes[node.id] = node

        self.last_inserted = node
        self.current_position = node.id

        # append values for visualization
        self.scores = np.append(self.scores, 0.0)
        self.p_kt_pred = np.append(self.p_kt_pred, 0.0)

    def normalize_probabilities_after_initialization(self):
        # Normalize probabilities accounting for reject or not
        if self.reject is not None:
            normalizer = sum([node.probability for node in self.nodes.values()]) + self.reject_probability
            for node in self.nodes.values():
                node.probability = node.probability/normalizer
            self.reject_probability = self.reject_probability/normalizer
        else:
            normalizer = sum([node.probability for node in self.nodes.values()])
            for node in self.nodes.values():
                node.probability = node.probability/normalizer

    # --- SIMILARITIES ---
    def top_k_nodes(self, similarities):
        sorted_indices = np.argsort(-similarities, axis=0)
        if len(sorted_indices) < 5:
            top_k = sorted_indices
        else:
            top_k = sorted_indices[:5]

        return top_k

    # --- PROBABILITIES (with LC) ---
    def localize(self, image_path, descriptor, bayesian = True):
        if bayesian:
            # Run Bayesian localization accounting for reject or not
            if self.reject is not None:
                if self.reject_strategy == 'non-localized':
                    # Reject node is considered as an additional localization with probability
                    self.p_kt_pred = self.run_prediction_reject()
                    chosen_id, self.most_similar, max_prob = self.update_probabilities_reject(descriptor, self.p_kt_pred)
                elif self.reject_strategy == 'diffusion':
                    # Reject node is used to skip update phase, but has it is not a possible localization
                    chosen_id, self.most_similar, max_prob = self.diffusion_reject(descriptor)

            else:
                self.p_kt_pred = self.run_prediction()
                chosen_id, self.most_similar, max_prob = self.update_probabilities(descriptor, self.p_kt_pred)

        else:
            # We retrieve the most similar node just using the descriptor
            chosen_id, self.most_similar, max_prob = self.localize_with_scores(descriptor)

        if (chosen_id != -1):
            loop_found = True
            loop_id = chosen_id
        else:
            loop_found = False
            loop_id = -1

        return loop_found, loop_id, max_prob

    def run_prediction(self, alpha = 0.95):
        # Predict probability -- no reject
        p_kt_pred = np.zeros(len(self.nodes.keys()))

        # Go through all nodes to sum probabilities under assumption of previous LC found 
        for id, node in self.nodes.items():
            # Equal probability for neighbors of node
            subgraph = nx.ego_graph(self.nx_graph, id, radius = self.radius_bayesian)
            neighbors = list(subgraph.nodes()) # get connected nodes IDs
            p_kt_pred[neighbors] += (alpha / len(neighbors)) * node.probability

            # Distribution of probability for rest of nodes
            rest_of_graph = [node.id for node in self.nodes.values() if node.id not in neighbors]
            p_kt_pred[rest_of_graph] += ((1-alpha) / len(rest_of_graph)) * node.probability

        return list(p_kt_pred)
    
    def run_prediction_reject(self):
        # Predict probability -- with reject
        p_kt_pred = np.zeros(len(self.nodes.keys()))

        # Probability under the assumption of no previous loop-closure
        self.reject_prediction = 0.5 * self.reject_probability # 0.7 for reject
        p_kt_pred += (0.5 / len(self.nodes)) * self.reject_probability # 0.3 shared between rest of nodes

        # Go through all nodes to sum probabilities under assumption of previous loop-closure found 
        for id, node in self.nodes.items():
            # Every node adds 0.1 to reject (no loop-closure) probability
            self.reject_prediction += 0.1 * node.probability # 0.1 for nonLC

            # Equal probability for neighbors of node
            subgraph = nx.ego_graph(self.nx_graph, id, radius = self.radius_bayesian)
            neighbors = list(subgraph.nodes()) # get connected nodes IDs
            p_kt_pred[neighbors] += (0.89 / len(neighbors)) * node.probability

            # Distribution of probability for rest of nodes
            rest_of_graph = [node.id for node in self.nodes.values() if node.id not in neighbors]
            p_kt_pred[rest_of_graph] += (0.01 / len(rest_of_graph)) * node.probability

        return list(p_kt_pred)

    def update_probabilities(self, descriptor, p_kt_pred):
        chosen_id = 0
        most_similar = -1

        similarities, scores = self.likelihood_estimator.estimate_similarities_topk(self, descriptor)

        self.saved_likelihood.append(list(similarities))
        self.scores = similarities

        # nodes probability
        p_kt = np.array(p_kt_pred) * similarities

        most_similar = np.argmax(similarities)

        np.clip(p_kt, 0.000001, 10000.0)
        normalizer = np.sum(p_kt)
        p_kt /=normalizer 

        for id, node in self.nodes.items():
            node.probability = p_kt[id]

        self.p_kt = [node.probability for node in self.nodes.values()]

        if len(self.nodes) > 2:
            # Get convolution of p_kt with kernel size 7
            #kernel = np.ones(self.radius_bayesian*2+1)
            kernel = np.ones(7)
            # print('KERNEL', kernel)
            aggregated_prob = np.convolve(p_kt, kernel, 'same')
            self.single_prob.append(list(self.p_kt))
            self.saved_probabilities.append(list(aggregated_prob))

            self.aggregated_p_kt = aggregated_prob
            max_value = np.amax(aggregated_prob)

            if max_value > self.threshold_probability:
                # Get the node with max probability
                self.found_loops.append(len(self.saved_probabilities)-1)
                chosen_id = np.argmax(aggregated_prob)

                # Extract scores window around max_value and get the max score
                scores_window = scores[chosen_id-self.radius_bayesian:chosen_id+self.radius_bayesian+1]
                max_score = np.amax(scores_window)
                
                # Find max score index in similiarities vector
                max_score_index = np.where(scores == max_score)[0][0]
                # if chosen_id != max_score_index:
                #     # If max_score is different from chosen_id, change the ID
                #     max_value = str(round(np.amax(max_value),2)) + ' - ' + str(chosen_id)
                #     chosen_id = max_score_index
            else: 
                chosen_id = -1
        else: 
            chosen_id = -1
        
        return chosen_id, most_similar, max_value
    
    def update_probabilities_reject(self, descriptor, p_kt_pred):
        chosen_id = 0
        most_similar = -1

        similarities, scores, reject_top_sim, reject_scores = self.likelihood_estimator.estimate_similarities_reject(self, descriptor)

        self.saved_likelihood.append(list(similarities))

        # nodes probability
        p_kt = np.array(p_kt_pred) * similarities
        
        # reject probability
        
        self.reject_probability = self.reject_prediction * reject_top_sim

        most_similar = np.argmax(similarities)
        normalizer = np.sum(p_kt) + self.reject_probability 

        np.clip(p_kt, 0.000001, 10000.0)
        normalizer = np.sum(p_kt)
        p_kt /=normalizer 
    
        for id, node in self.nodes.items():
            node.probability = p_kt[id]

        self.reject_probability /= normalizer
        self.p_kt = [node.probability for node in self.nodes.values()]

        self.scores = similarities
        self.reject_scores = reject_scores
        self.reject_likelihood = reject_top_sim
        
        # TODO: compare against reject
        if len(self.nodes) > 2:
            # Get convolution of p_kt with kernel size 7
            kernel = np.ones(self.radius_bayesian*2+1)
            aggregated_prob = np.convolve(p_kt, kernel, 'same')
            self.single_prob.append(list(self.p_kt))
            self.saved_probabilities.append(list(aggregated_prob))

            self.aggregated_p_kt = aggregated_prob

            max_value = np.amax(aggregated_prob)
            print(f'MAX PROBABILITY FOUND {max_value}')
            if max_value > 0.9:
                self.found_loops.append(len(self.saved_probabilities)-1)
                chosen_id = np.argmax(aggregated_prob)
                # Extract scores window around max_value and get the max score
                scores_window = scores[chosen_id-self.radius_bayesian:chosen_id+self.radius_bayesian+1]
                max_score = np.amax(scores_window)
                print(f'MAX SCORE FOUND {max_score}')
                # Find max score index in similiarities vector
                max_score_index = np.where(scores == max_score)[0][0]
                if chosen_id != max_score_index:
                    print(f'INITIAL LOOP ID {chosen_id}')
                    chosen_id = max_score_index
            else: 
                chosen_id = -1
        else: 
            chosen_id = -1
        
        return chosen_id, most_similar, max_value
    
    def diffusion_reject(self, descriptor):
        # Diffuse probabilities if current image is reject
        similarities, reject_scores, is_reject = self.likelihood_estimator.image_is_reject(self, descriptor)
        most_similar = np.argmax(similarities)
        
        self.reject_scores = reject_scores
        self.is_reject = is_reject

        # If image is reject, just do prediction (diffuse probability)
        if is_reject:
            p_kt_pred = self.run_prediction(alpha = 0.95)
            for id, node in self.nodes.items():
                node.probability = p_kt_pred[id]

            self.p_kt = [node.probability for node in self.nodes.values()]

            kernel = np.ones(self.radius_bayesian*2+1)
            aggregated_prob = np.convolve(p_kt_pred, kernel, 'same')

            # Save probabilities
            self.single_prob.append(list(self.p_kt))
            self.saved_probabilities.append(list(aggregated_prob))
            self.saved_likelihood.append(list(similarities))
            self.scores = similarities

            self.aggregated_p_kt = aggregated_prob

            max_value = str(round(np.amax(aggregated_prob),2)) + ' - reject'
            chosen_id = -1
            
        else:
            # If image is not reject, do prediction and update probabilities
            p_kt_pred = self.run_prediction()
            chosen_id, most_similar, max_value = self.update_probabilities(descriptor, p_kt_pred)
            
        return chosen_id, most_similar, max_value

    def localize_with_scores(self, descriptor):
        _, scores, _, reject_scores = self.likelihood_estimator.estimate_similarities_reject(self, descriptor)
        self.scores = scores
        self.reject_scores = reject_scores
        self.saved_likelihood.append(list(scores))

        # Get max value and index for scores and reject_scores
        max_score = np.amax(scores)
        chosen_id = np.argmax(scores)
        most_similar = chosen_id

        max_reject_score = np.amax(reject_scores)

        if self.reject_outliers is True:
            if max_reject_score > max_score:
                chosen_id = -1
                most_similar = -1
                max_score = max_reject_score
        
        return chosen_id, most_similar, max_score

    # --- SAVING FUNCTIONS ---
    def save_graph(self):
        graph_pkl = os.path.join(self.save_path, 'graph.pkl')
        ids = []
        n_frames = []
        images = []
        descriptors = []
        for node_id, node in self.nodes.items():
            ids.append(node_id)
            n_frames.append(node.n_frame)
            images.append(node.image_paths)
            descriptors.append(node.descriptor)

        print('>> {}: Saving graph pkl...')

        data = {'node_ids': ids, 'n_frames': n_frames, 'images': images, 'descriptors': descriptors}

        with open(graph_pkl, "wb") as f:
            pickle.dump(data, f)

    def save_results_text_endomapper(self, results, dataset, save_folder=None, experiment_name=None):
        if experiment_name is not None:
            if not exists(join(save_folder, experiment_name)):
                os.makedirs(join(save_folder, experiment_name))
            if not exists(join(save_folder, experiment_name, dataset)):
                os.makedirs(join(save_folder, experiment_name, dataset))
            results_file = join(save_folder, experiment_name, dataset, dataset + '_localizations.txt')
        else:
            results_file = join(self.save_path, dataset + '_localizations.txt')
        
        # Write results to file
        # Format: frame_id - node_id
        with open(results_file, 'w') as f:
            for frame_id, results in results.items():
                node_id = results[0]
                score = results[1]
                # Check if score is str
                if isinstance(score, str):
                    f.write(str(frame_id).zfill(4) + ': ' + str(node_id) + ' - ' + score + '\n')
                else:
                    f.write(str(frame_id).zfill(4) + ': ' + str(node_id) + ' - ' + str(round(score, 3)) + '\n')

    def save_results_text_c3vd(self, results, sequence, save_folder=None, experiment_name=None):
        if experiment_name is not None:
            if not exists(join(save_folder, experiment_name)):
                os.makedirs(join(save_folder, experiment_name))
            if not exists(join(save_folder, experiment_name, 'C3VD')):
                os.makedirs(join(save_folder, experiment_name, 'C3VD'))
            results_file = join(save_folder, experiment_name, 'C3VD', sequence + '_localizations.txt')
        else:
            if not exists(join(save_folder, 'C3VD')):
                os.makedirs(join(save_folder, 'C3VD'))
            results_file = join(self.save_path, 'C3VD', sequence + '_localizations.txt')
        
        # Write results to file
        # Format: frame_id - node_id
        with open(results_file, 'w') as f:
            for frame_id, results in results.items():
                node_id = results[0]
                score = results[1]
                # Check if score is str
                if isinstance(score, str):
                    f.write(str(frame_id*5).zfill(4) + ': ' + str(node_id) + ' - ' + score + '\n')
                else:
                    f.write(str(frame_id*5).zfill(4) + ': ' + str(node_id) + ' - ' + str(round(score, 3)) + '\n')

    def save_results(self, results, dataset):
        results_pkl = os.path.join(self.save_path, dataset + '_localizations.pkl')
        ids = []
        n_frames = []
        images = []
        for n_frame, loop_pair in results.items():
            n_frames.append(n_frame)
            image_list = loop_pair[0] + loop_pair[1]
            images.append(image_list)

        for node_id, node in self.nodes.items():
            ids.append(node_id)
            n_frames.append(node.n_frame)
            images.append(node.image_paths)

        print('>> {}: Saving results pkl...')

        data = {'n_frames': n_frames, 'images': images}

        with open(results_pkl, "wb") as f:
            pickle.dump(data, f)

    def save_probabilities(self):
        graph_pkl = os.path.join(self.save_path, 'supplementary.pkl')

        print('>> {}: Saving supplementary pkl...')

        data = {'probabilities': self.saved_probabilities, 'likelihood': self.saved_likelihood, 'single_prob': self.single_prob}

        with open(graph_pkl, "wb") as f:
            pickle.dump(data, f)


