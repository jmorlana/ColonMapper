from audioop import mul
from numpy import empty
import torch
import numpy as np

class Node:
    def __init__(self, descriptor, image_path, n_frame, id, probability = 0.0):
        self.descriptor = descriptor
        self.image_path = image_path
        self.n_frame = n_frame
        self.id = id
        self.probability = probability
        self.edges = []

        self.image = None
        self.keypoints_data = None

    def add_more_views(self, new_descriptor):
        self.descriptor = torch.cat((self.descriptor, new_descriptor), dim = 1)

class MultiNode:
    def __init__(self, descriptor, image_paths, n_frame, id, probability = 0.0):
        self.descriptor = descriptor
        if isinstance(image_paths, list):
            self.image_paths = image_paths
            self.frames_in_node = len(image_paths)
        else:
            self.image_paths = [image_paths]
            self.frames_in_node = 1
        self.n_frame = n_frame
        self.id = id
        self.probability = probability
        self.edges = []
        self.image = None
        self.keypoints_data = None

    def add_more_views(self, new_descriptor, image):
        self.descriptor = torch.cat((self.descriptor, new_descriptor), dim = 1)
        if isinstance(image, list):
            self.image_paths.extend(image)
            self.frames_in_node += len(image)
        else:
            self.image_paths.append(image)
            self.frames_in_node += 1

    def number_of_frames(self):
        return self.frames_in_node