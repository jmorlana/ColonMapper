"""
Main functions for mapping and localization using topological maps.
"""

from datetime import datetime
import logging
import numpy as np
import torch
import cv2

import time
from datasets.genericdataset import ImagesFromList
from datasets.vg_genericdataset import ImagesFromListVG
from .node import MultiNode
from .topological_map_mapping import TopologicalMapMapping
from .topological_map_localization import TopologicalMapLocalization
from .visualization_utils import gui_plot_mapping
from .gui_plotter import GUIPlotter
from .likelihood import LikelihoodEstimator
import matplotlib.pyplot as plt
import os


def topological_mapping(conf, net, images, image_size, output_dim, transform, vg=False, plot=False, ms=[1], msp=1):
    # creating dataset loader
    if vg == True:
        loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=images, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
    else:
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=image_size, transform=transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )

    likelihood_estimator = LikelihoodEstimator(vg, top_k=7, short_sim=conf['short_sim'], loftr_conf=0.4)

    # create graph and get conf values
    graph = TopologicalMapMapping(conf, likelihood_estimator=likelihood_estimator, vg=vg)
    max_skipped = conf['max_skipped']
    matching_node = conf['matching_node']
     
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # Create proto-node and fill it with new images
    proto_node = None
    skipped = 0

    # extracting vectors
    with torch.no_grad():
        last_inserted = None 
        id = 0     
        state = '' 

        node_closed = False
        last_node_update = 0
        update_allowed = True
        reject = []

        for i, input in enumerate(loader):
            input = input.cuda()

            # Extract descriptor
            descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
            descriptor = descriptor.view(descriptor.shape[0], 1)

            # Avoid consecutive updates -- they tend to be walls / water images
            update_allowed = (i - last_node_update) > 3

            reject.append(images[i])
            
            # Initialize proto-node. This happens after closing a node (discarded or added to map)
            if proto_node is None:
                n_frame = os.path.basename(images[i]).replace('.png', '').zfill(5)
                proto_node = MultiNode(descriptor, images[i], n_frame, id, probability = 0.0)
                state = 'START NODE'
                last_node_update = i
                print(i, ': ', state)
                gui_plot_mapping(graph, images[i], state, proto_node, plot=False)
                continue

            else:
                # Start filling proto-node with new images
                short_term_similarity, similarities = likelihood_estimator.check_short_term_similarity_proto_node(proto_node, descriptor)

                if (short_term_similarity):
                    # Images are too similar so we skip
                    state = 'SIMILAR - SKIP'
                    skipped += 1
                    # If we skip too much, add image
                    if skipped >= max_skipped:
                        skipped = 0
                        matching = likelihood_estimator.check_medium_term_similarity_proto_node(graph, proto_node, images[i], matching_node=matching_node)
                        if matching:
                            state = 'UPDATE NODE'
                            proto_node.add_more_views(descriptor, images[i])
                            last_node_update = i
                            # plot_proto_node(proto_node, similarities, plot = plot)
                        else:
                            node_closed = True
                elif (likelihood_estimator.check_medium_term_similarity_proto_node(graph, proto_node, images[i], matching_node=matching_node) and update_allowed):
                    # Image is a new view from the current place
                    state = 'UPDATE NODE'
                    proto_node.add_more_views(descriptor, images[i])
                    last_node_update = i
                    skipped = 0
                    #plot_proto_node(proto_node, similarities, plot = plot)
                else:
                    node_closed = True
                    skipped = 0
                
                # Close node if it has more than 9 views
                if (proto_node.number_of_frames() > 9):
                    node_closed = True
                    skipped = 0

                # If node has been closed, check if node is valid and then localize
                if node_closed:
                    # Reset state for next iteration
                    node_closed = False

                    # Proto-node is finished
                    if (proto_node.number_of_frames() < 3):
                        # Node is discarded as it has very few views
                        state = 'RESET NODE'
                        # Discard previous proto-node 
                        proto_node = None
                        
                        # Append to reject the discarded frames
                        graph.reject.append(reject)
                        reject = []
                    else:
                        # Append reject to mapped_frames as images are building a node
                        graph.mapped_frames.append(reject)
                        reject = []
                        # Insert the node if it is the first one
                        if last_inserted is None:
                            # First node
                            state = 'NEW NODE'
                            next_id = id + 1

                            graph.add_protonode(proto_node)
                            last_inserted = graph.last_inserted
                            id = next_id

                            proto_node = None

                        else:
                            # Graph is already initialized, add new node 
                            state = 'NEW NODE'
                            next_id = id + 1
                            graph.add_protonode(proto_node, graph.current_position)
                            print(f'Added new node{proto_node.id} with {len(proto_node.image_paths)} views.')

                            last_inserted = graph.last_inserted
                            id = next_id

                            gui_plot_mapping(graph, images[i], state, proto_node, plot=False)
                            proto_node = None
   
            gui_plot_mapping(graph, images[i], state, proto_node, plot=True)
            print(i, ': ', state)
            # plot_current_frame(images[i], state, plot = plot)
    #cv2.waitKey()
    return graph

def bayesian_localization_images(conf, graph_dict, net, images, reject, image_size, output_dim, transform, vg=False, plot=False, bayesian=True, reload_graph=False, ms=[1], msp=1):
    # Get 1 image out of conf['multiple_to_localize']
    images = images[::conf['multiple_to_localize']]

    # creating dataset loader
    if vg == True:
        loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=images, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
    else:
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=image_size, transform=transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )

    gui = GUIPlotter(len(graph_dict['node_ids']))
    likelihood_estimator = LikelihoodEstimator(vg, method=conf['likelihood'], top_k=7, short_sim=conf['short_sim'], threshold=conf['threshold_likelihood'])
    # create graph
    graph = TopologicalMapLocalization(conf, likelihood_estimator=likelihood_estimator, vg=vg, reject=reject, width=gui.width, height=gui.height)
    if reload_graph:
        logging.info('Reloading graph descriptors...')
        graph_dict = reload_graph_descriptors(graph_dict, net, image_size, output_dim, transform, vg, ms, msp)
    graph.initialize_graph(graph_dict, conf['initial_node'])

    if reject is not None:
        graph.reject_descriptors = extract_reject_descriptors(net, reject, image_size, output_dim, transform, vg, ms, msp)
        gui.read_reject_images(reject)

    node_images_original = [node.image for node in graph.nodes.values()]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    results = {}

    # extracting vectors
    start_time = datetime.now()
    with torch.no_grad():
        state = ''
        # Create proto-node just for empty visualization
        proto_node = None

        for i, input in enumerate(loader):
            input = input.cuda()

            # Extract descriptor
            descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)

            loop_id = -1

            # Localize every incoming image
            loop_found, loop_id, max_prob = graph.localize(images[i], descriptor, bayesian)

            if loop_found:
                # Check if max_prob is a string
                if isinstance(max_prob, str):
                    state = f'LOCALIZED - {max_prob} with node {loop_id}'
                else:
                    state = f'LOCALIZED - {max_prob:.2f} with node {loop_id}'
                #print(state, 'with node', loop_id)
                #results[i] = [graph.nodes[loop_id].image_paths, proto_node.image_paths]
                results[i] = [loop_id, max_prob]
            else:
                # Check if max_prob is a string
                if isinstance(max_prob, str):
                    state = f'NOT LOCALIZED - {max_prob}'
                else:
                    state = f'NOT LOCALIZED - {max_prob:.2f}'
                results[i] = ['N', max_prob]

            gui.gui_plot_localization(graph, node_images_original, loop_id, images[i], state, i, plot = True, wait=200, experiment_folder = conf['experiment_folder'])
            print(i, ': ', state)
    logging.info(f"Localization time {str(datetime.now() - start_time)[:-7]} for {len(images)} images.")

    return graph, results

def reload_graph_descriptors(graph_dict, net, image_size, output_dim, transform, vg=False, ms=[1], msp=1):
    likelihood_estimator = LikelihoodEstimator(vg)
    start_time = datetime.now()
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    new_graph_descriptors = []
    for node_images in graph_dict['images']:
        # creating dataset loader
        if vg == True:
            loader = torch.utils.data.DataLoader(
                    ImagesFromListVG(root='', images=node_images, imsize=image_size, transform=transform),
                    batch_size=1, shuffle=False, num_workers=8, pin_memory=True
                )
        else:
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=node_images, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )

        # Initialize torch tensor for descriptors DxN
        descriptors = torch.zeros((output_dim, len(node_images)))
        for i, input in enumerate(loader):
            input = input.cuda()

            # Extract descriptor
            descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
            descriptors[:,i] = descriptor

        new_graph_descriptors.append(descriptors)

    graph_dict['descriptors'] = new_graph_descriptors
    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
        
    return graph_dict

def extract_reject_descriptors(net, reject, image_size, output_dim, transform, vg=False, ms=[1], msp=1):
    # creating dataset loader
    if vg == True:
        loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=reject, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
    else:
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=reject, imsize=image_size, transform=transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )

    likelihood_estimator = LikelihoodEstimator(vg, top_k=7, short_sim=0.60)

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # extracting vectors
    with torch.no_grad():
        # Initialize torch tensor for descriptors DxN
        descriptors = torch.zeros((output_dim, len(reject)))
        for i, input in enumerate(loader):
            input = input.cuda()

            # Extract descriptor
            descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)
            descriptors[:,i] = descriptor

    return descriptors

def reject_visualization(conf, graph_dict, net, images, image_size, output_dim, transform, vg=False, ms=[1], msp=1):
    # Extract reject from dict
    reject = graph_dict['reject']
    images = [frame[-1] for frame in reject]
     # creating dataset loader
    if vg == True:
        loader = torch.utils.data.DataLoader(
                ImagesFromListVG(root='', images=images, imsize=image_size, transform=transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
    else:
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=images, imsize=image_size, transform=transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )

    likelihood_estimator = LikelihoodEstimator(vg, top_k=7, short_sim=0.60)
    # create graph
    graph = TopologicalMapLocalization(conf, output_dim, likelihood_estimator=likelihood_estimator, vg=vg)
    graph.initialize_graph(graph_dict, conf['initial_node'])

    node_images_original = [node.image for node in graph.nodes.values()]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    results = {}
    plot = True

    # Plot node number 10
    image_paths = graph.nodes[10].image_paths
    node_images = [cv2.imread(image_path) for image_path in image_paths]
    node_images = [cv2.resize(image, (300, 250)) for image in node_images]
    cv2.imshow('Node 10', cv2.hconcat(node_images))
    cv2.waitKey(0)

    # extracting vectors
    with torch.no_grad():
        dataloader_iterator = iter(loader)
        input = next(dataloader_iterator)
        input = input.cuda()  

        for i, input in enumerate(loader):
            input = input.cuda()

            # Extract descriptor
            descriptor = likelihood_estimator.extract_descriptor(net, input, ms, msp)

            similarities = likelihood_estimator.estimate_similarities(graph, descriptor.view(descriptor.shape[0], 1))

            # Find index with highest similarity using argmax
            max_index = np.argmax(similarities)
            max_value = similarities[max_index]
            node_image = node_images_original[max_index]

            ## Plot similarities, node image and current image
            #node_image_cv = cv2.imread(node_image)
            node_image_cv = cv2.resize(node_image, (500, 400))
            current_image_cv = cv2.imread(images[i])
            current_image_cv = cv2.resize(current_image_cv, (500, 400))
            cv2.putText(node_image_cv, f'Node {max_index} - {max_value:.2f}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 10), 2)
            cv2.putText(current_image_cv, f'Current image', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 10), 2)

            # Get frame number from basename
            frame_number = int(os.path.basename(images[i]).split('.')[0])
            cv2.putText(current_image_cv, str(frame_number), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 255, 10), 1)

            # Concatenate images
            img_v = cv2.vconcat([node_image_cv, current_image_cv])
            
            # Create a figure for plotting similarities
            if i > 0:
                plt.clf()
            else:
                plt.figure(figsize=(10, 5))
            plt.bar(np.arange(len(similarities)), similarities)
            plt.title('Similarities')
            plt.xlabel('Node')
            plt.ylabel('Similarity')
            
            plt.pause(0.5)
            plt.plot()

            cv2.imshow('reject', img_v)
            cv2.waitKey(0)

    return None