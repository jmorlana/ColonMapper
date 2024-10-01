from matplotlib.pyplot import plot
import torch
import cv2
import numpy as np
import sys
from pathlib import Path
import matplotlib.cm as cm
import kornia as K
import kornia.feature as KF
import os.path
from settings import EVAL_PATH

from cirtorch_networks.imageretrievalnet import extract_ss, extract_ms
from .visualization_utils import plot_LoFTR_matches

def load_torch_image(fname):
    img = cv2.imread(fname) 
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    img_tensor = K.image_to_tensor(img, False).float() /255.
    img_tensor = K.color.bgr_to_rgb(img_tensor)
    return img, img_tensor

class LikelihoodEstimator:
    def __init__(self, vg=False, method = 'topk',top_k = 7, short_sim = 0.95, threshold =0.5, loftr_conf = 0.4, backbone = 'cct384'):
        self.vg = vg
        self.top_k = top_k
        self.short_similarity = short_sim
        self.loftr_conf = loftr_conf
        self.method = method
        self.threshold = threshold
        self.debug_folder = EVAL_PATH / 'loftr_logs'
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        if threshold == 0.5:
            self.low_value = 0.3
        else:
            self.low_value = 0.5
        if backbone == 'cct384':
            self.max_distance = 0.40
        elif backbone == 'resnet50':
            self.max_distance = 1.5

    def extract_descriptor(self, net, image, ms, msp):
        if self.vg:
            # Extract descriptor as a column vector and normalize it
            descriptor = net(image).permute(1,0)
            descriptor = descriptor / descriptor.norm(dim=0)
            descriptor = descriptor.cpu().data.squeeze()
            return descriptor
        else:
            if len(ms) == 1 and ms[0] == 1:
                descriptor = extract_ss(net, image)
            else:
                descriptor = extract_ms(net, image, ms, msp)
            return descriptor

    def global_descriptor_similarity(self, incoming_descriptor, node_descriptor):
        # Compare a pair of descriptors.
        return np.dot(incoming_descriptor.numpy().T, node_descriptor.numpy())

    def check_short_term_similarity(self, graph, descriptor):
        last_node = graph.nodes[graph.current_position]

        # Check how many images are closer than threshold
        similarity = self.global_descriptor_similarity(descriptor, last_node.descriptor) > self.short_similarity
        votes = np.sum(similarity, axis=0)

        if votes > 0:
            return True
        else:
            return False  

    def check_short_term_similarity_proto_node(self, proto_node, descriptor):
        similarities = self.global_descriptor_similarity(descriptor, proto_node.descriptor)
        #print('Similarity', similarities)
        max_value = np.max(similarities)
        similarity = similarities > self.short_similarity
        votes = np.sum(similarity, axis=0)

        if votes.any():
            return True, similarities
        else:
            return False, similarities

    def check_medium_term_similarity_proto_node(self, graph, proto_node, image, matching_node = 'last'):
        # if proto_node.number_of_frames() >15:
        #     return False

        img1_np, img1 = load_torch_image(image)
        if matching_node == 'last':
            img2_np, img2 = load_torch_image(proto_node.image_paths[-1])
        elif matching_node == 'half':
            idx = int(len(proto_node.image_paths)/2)
            img2_np, img2 = load_torch_image(proto_node.image_paths[idx])
        

        if graph.mask is not None:
            input_dict = {"image0": K.color.rgb_to_grayscale(img1.to(graph.device)), # LofTR works on grayscale images only 
                        "image1": K.color.rgb_to_grayscale(img2.to(graph.device)),
                        "mask0": graph.mask, "mask1": graph.mask}
        else:
            input_dict = {"image0": K.color.rgb_to_grayscale(img1.to(graph.device)), # LofTR works on grayscale images only 
                        "image1": K.color.rgb_to_grayscale(img2.to(graph.device))}

        with torch.inference_mode():
            correspondences = graph.matcher(input_dict)

        kpts0 = correspondences['keypoints0'].cpu().numpy()
        kpts1 = correspondences['keypoints1'].cpu().numpy()
        inliers = (correspondences['confidence'] > self.loftr_conf).cpu().numpy()

        mkpts0 = kpts0[inliers]
        mkpts1 = kpts1[inliers]
        mconf = correspondences['confidence'][inliers].cpu().numpy()

        success = mkpts0.shape[0] > 100
        if success: 
            text_result = 'SUCCESS'
        else: 
            text_result = 'FAILURE'

        frame_id = graph.saved_frames
        graph.saved_frames += 1

        filename = os.path.join(self.debug_folder, str(frame_id).zfill(5) + '.png')
        if graph.do_viz:
            # Visualize the matches.
            color = cm.jet(mconf)
            text = [
                'Keypoints: {}:{}'.format(len(mkpts0), len(mkpts1)),
                'Matches: {}'.format(len(mkpts0)),
                text_result,
            ]
            
            plot_LoFTR_matches(
                img1_np, img2_np, kpts0, kpts1, mkpts0, mkpts1, color,
                text, path =filename, show_keypoints=False,
                opencv_display=False, opencv_title='LoFTR Matches')
        
        if success:
            return True
        else:
            return False

    def estimate_similarities(self, graph, descriptor):
        similarities = np.zeros(len(graph.nodes))

        node_ids = [node_id for node_id, node in graph.nodes.items() for _ in range(node.descriptor.shape[1])]
        node_ids = np.array(node_ids)

        for id, node in graph.nodes.items():
            similarity = self.global_descriptor_similarity(descriptor, node.descriptor)
            similarities[id] = np.max(similarity)

        return similarities

    def estimate_similarities_topk(self, graph, descriptor):
        top_k = self.top_k
        similarities = np.zeros(len(graph.nodes))
        topk_sim = np.zeros(len(graph.nodes)) + 0.2

        node_ids = [node_id for node_id, node in graph.nodes.items() for _ in range(node.descriptor.shape[1])]
        node_ids = np.array(node_ids)

        for id, node in graph.nodes.items():
            similarity = self.global_descriptor_similarity(descriptor, node.descriptor)
            similarities[id] = np.max(similarity)

        ranks = np.argsort(-similarities, axis=0)[:top_k]
        first_tier = int(top_k/3)
        second_tier = int(2*top_k/3)

        # If similarities[rank] < threshold, set topk_sim to a small value
        topk_sim[ranks[similarities[ranks] < self.threshold]] = self.low_value
        topk_sim[ranks[similarities[ranks] >= self.threshold]] = similarities[ranks[similarities[ranks] >= self.threshold]]

        return topk_sim, similarities
    
    def estimate_similarities_reject(self, graph, descriptor):
        similarities = np.zeros(len(graph.nodes))

        node_ids = [node_id for node_id, node in graph.nodes.items() for _ in range(node.descriptor.shape[1])]
        node_ids = np.array(node_ids)
        
        # Similarity against map nodes
        for id, node in graph.nodes.items():
            similarity = self.global_descriptor_similarity(descriptor, node.descriptor)
            similarities[id] = np.max(similarity)

        # Similarity against reject 
        reject_similarity = self.global_descriptor_similarity(descriptor, graph.reject_descriptors)

        if self.method == 'topk':
            return self.topk_scores(graph, similarities, reject_similarity)
        elif self.method == 'mean_std':
            return self.mean_std_scores(similarities, reject_similarity)        

    def topk_scores(self, graph, nodes_similarities, reject_similarities=None):
        topk_sim = np.zeros(len(graph.nodes)) + 0.2
        # Rank map nodes 
        ranks = np.argsort(-nodes_similarities, axis=0)[:self.top_k]
        
        if reject_similarities is None:
            topk_reject = None
        else:
            # Rank reject and average top results
            reject_rank = np.argsort(-reject_similarities, axis=0)[:3]
            topk_reject = np.mean(reject_similarities[reject_rank])

            # Check if reject score is better than any of the map nodes
            if np.any(topk_reject > nodes_similarities[ranks]):
                # Discard the last rank value
                ranks = ranks[:-1]
            else:
                # Saturated reject score
                topk_reject = 0.2

        # Get the top ranked similarities
        topk_sim[ranks] = nodes_similarities[ranks]
        
        return topk_sim, nodes_similarities, topk_reject, reject_similarities
    
    def mean_std_scores(self, nodes_similarities, reject_similarities=None):
        # Get mean and std of map nodes
        mean = np.mean(nodes_similarities)
        std = np.std(nodes_similarities)

        # Compute score: if similarity > mean + std, score = (sim-std)/mean, else score = 1
        nodes_likelihood = np.where(nodes_similarities > mean + std, 1.5*(nodes_similarities - std) / mean, 1)

        if reject_similarities is None:
            reject_likelihood = None
        else:
            reject_likelihood = np.where(reject_similarities > mean + std, (reject_similarities - std) / mean, 1)
            reject_likelihood = np.mean(reject_likelihood) # TODO: play with this

        return nodes_likelihood, nodes_similarities, reject_likelihood, reject_similarities
    
    def image_is_reject(self, graph, descriptor):
        similarities = np.zeros(len(graph.nodes))

        node_ids = [node_id for node_id, node in graph.nodes.items() for _ in range(node.descriptor.shape[1])]
        node_ids = np.array(node_ids)
        
        for id, node in graph.nodes.items():
            similarity = self.global_descriptor_similarity(descriptor, node.descriptor)
            similarities[id] = np.max(similarity)

        # Similarity against reject 
        reject_similarity = self.global_descriptor_similarity(descriptor, graph.reject_descriptors)

        # Compare if the 3 top-ranked reject nodes are better than map nodes
        reject_rank = np.argsort(-reject_similarity, axis=0)[:3]
        reject_score = np.mean(reject_similarity[reject_rank])
        map_score = np.mean(similarities[np.argsort(-similarities, axis=0)[:3]])


        if reject_score > map_score:
            # print(f'reject score: {reject_score:.2f}  -  Map score: {map_score:.2f}' )
            is_reject = True
        else:
            is_reject = False

        # Get top-k scores
        topk_sim = np.zeros(len(graph.nodes)) + 0.2
        ranks = np.argsort(-similarities, axis=0)[:self.top_k]
        topk_sim[ranks] = similarities[ranks]

        return topk_sim, reject_similarity, is_reject