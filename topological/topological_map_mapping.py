from collections import OrderedDict
import numpy as np
import networkx as nx
import os.path
import pickle

import cv2
import kornia as K
import kornia.feature as KF
from settings import root

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class TopologicalMapMapping:
    def __init__(self, conf, likelihood_estimator, vg=False):
        self.radius_bayesian = conf['radius_bayesian']
        self.save_path = conf['save_path']

        self.last_image_inserted = 0

        self.nx_graph = nx.Graph()
        self.nodes = OrderedDict()
        self.n_connections = OrderedDict()
        self.current_position = 0
        self.last_inserted = None

        self.default_image = cv2.resize(cv2.imread(str(root / 'assets/default.jpg')), (250, 200))
        self.first_plot = True
        self.read_image = True
        self.most_similar = -1
        self.p_kt_pred = []
        self.p_kt = []
        self.scores = []
        self.scores_sg = []

        self.device = conf['verification']['device']
        self.resize = conf['verification']['resize']
        self.sg_matches = None

        self.mask = self.load_mask()

        # Last inserted/updated nodes.
        self.last_nodes = []
        self.init_probabilities = True

        self.saved_frames = 0
        self.GeM_times = []

        # verification
        if conf['verification'] is not None:
            self.matcher = KF.LoFTR(pretrained='outdoor').eval().to(self.device)
            self.verification = True
            self.do_viz = False
        else:
            self.matcher = None
            self.verification = False
            self.do_viz = False

        self.likelihood_estimator = likelihood_estimator
        self.mapped_frames = []
        self.reject = []

    def load_mask(self):
        img = cv2.imread(str(root / 'assets/mask.png'), cv2.IMREAD_GRAYSCALE)
        mask = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((3, 3), np.uint8)  # change the kernel size as needed
        mask = cv2.erode(mask, kernel, iterations=15)
        mask_tensor = K.image_to_tensor(mask, False).float()
        return mask_tensor.squeeze(0).to(self.device)

    def add_protonode(self, proto_node, node_to_connect = -1):
        node = proto_node

        print('Nodes connected', node.id, '->', node_to_connect)
        if self.read_image:
            node.image = cv2.imread(node.image_paths[0])
            node.image = cv2.resize(node.image, (250, 200))
            node_text = str(node.id) + ' - ' + str(len(node.image_paths))
            cv2.putText(node.image, node_text, (20, 50), font, fontScale, (10, 255, 10), lineType)
            cv2.putText(node.image, str(node.n_frame), (20, 80), font, 1, (10, 255, 10), 1)

        if (node_to_connect != -1):
            self.nx_graph.add_node(node.id)
            self.nx_graph.add_edge(self.nodes[node_to_connect].id, node.id)

            self.nodes[node.id] = node
            self.update_connectivity(node.id)

        else:
            # First node doesn't have initial edges
            self.nx_graph.add_node(node.id)
            self.nodes[node.id] = node
            self.n_connections[node.id] = 0

        self.last_inserted = node
        self.current_position = node.id

    def update_connectivity(self, id):
        subgraph = nx.ego_graph(self.nx_graph, id, radius = self.radius_bayesian)
        neighbors = list(subgraph.nodes())
        neighbors.remove(id)

        self.n_connections[id] = len(neighbors)

        for neigh in neighbors:
            subgraph = nx.ego_graph(self.nx_graph, neigh, radius = self.radius_bayesian)
            curr_neighbors = list(subgraph.nodes())
            curr_neighbors.remove(neigh)
            self.n_connections[neigh] = len(curr_neighbors)

    def save_graph(self):
        graph_pkl = os.path.join(self.save_path)
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

        if len(self.mapped_frames) > 0:
            data = {'node_ids': ids, 'n_frames': n_frames, 'images': images, 'descriptors': descriptors, 
                    'mapped_frames': self.mapped_frames, 'reject': self.reject}
        else:
            data = {'node_ids': ids, 'n_frames': n_frames, 'images': images, 'descriptors': descriptors}

        with open(graph_pkl, "wb") as f:
            pickle.dump(data, f)

    





