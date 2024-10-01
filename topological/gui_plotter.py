import cv2
import numpy as np
import math
import os
from os.path import join
import shutil
from settings import root, DATA_PATH, EVAL_PATH

from utils.point_matrix import positions, regions, mapping

from utils.evaluate_endomapper import read_grund_truth_labels, read_node_labels

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

class GUIPlotter:
    def __init__(self, num_images = 100, max_width = 3250, max_height = 1800, frame_size = (300,250)):
        self.num_images = num_images
        self.max_width = max_width
        self.max_height = max_height
        self.frame_size = frame_size

        self.rows, self.cols, self.width, self.height = self.calculate_best_grid()

        # Load default images
        self.colon_img = cv2.imread(str(root / 'assets/colon_background.png'))
        self.default_image = cv2.resize(cv2.imread(str(root / 'assets/default.jpg')), frame_size)
        self.default_image_mapping = cv2.resize(cv2.imread(str(root / 'assets/default.jpg')), (self.width, self.height))
        self.reject_images = []
        self.previous_frames = []

        self.gt = read_grund_truth_labels(str(DATA_PATH / 'endomapper/eval_cross.txt'))
        self.map_labels = read_node_labels(str(DATA_PATH / 'endomapper/canonical_maps/labels_cross.txt'))
        save_localization_path = EVAL_PATH / 'localization_frames'
        if not save_localization_path.exists():
            os.makedirs(save_localization_path)
        else:
            # Clear all images in the folder
            shutil.rmtree(save_localization_path)
            os.makedirs(save_localization_path)
        self.save_localization_path = str(save_localization_path)


    def calculate_best_grid(self, init_cols=13, init_width=250, init_height=200, aspect_ratio=1.25):
        # Do initial estimation
        height = init_height
        width = init_width
        cols = init_cols
        rows = math.ceil(self.num_images / cols)

        height_occupied = rows * height
        if height_occupied > self.max_height:
            print("The initial grid does not fit in the given height")
            # Reduce size aggresively
            while height_occupied > self.max_height:
                height = height - 10
                width = height * aspect_ratio
                height_occupied = rows * height

        while (height_occupied < self.max_height):
            # Do another estimation by increasing the size of the images
            width = width + 10
            height = width / aspect_ratio

            # Check if with the current size we need to reduce cols
            width_occupied = cols * width
            while width_occupied > self.max_width:
                cols = cols - 1
                width_occupied = cols * width

            # Calculate the number of rows
            rows = math.ceil(self.num_images / cols)
            height_occupied = rows * height

        if height_occupied < self.max_height:
            height = height + (self.max_height - height_occupied) / rows
            width = height * aspect_ratio

        return rows, cols, int(width), int(height)
    
    def read_reject_images(self, reject_images):
        for image in reject_images:
            self.reject_images.append(cv2.resize(cv2.imread(image), (self.width, self.height)))

    def append_previous_frames(self, frame):
        # Append frame to previous frames, and remove the first one if the list is full (size = 4)
        self.previous_frames.append(frame)
        if len(self.previous_frames) > 4:
            self.previous_frames.pop(0)
        
    def get_result(self, image_number, loop_id):
        gt_label = self.gt[image_number]
        if gt_label.startswith('N'):
            return 'NONE'
        
        if loop_id == -1:
            return 'NOT LOCALIZED'
        else:
            node_label = self.map_labels[loop_id]
        
        if gt_label == node_label:
            return 'CORRECT'
        else:
            return 'WRONG'

    def gui_plot_localization(self, graph, images_original, loop_id, current_image, state, image_number, plot=False, wait = 100, experiment_folder = None):
        if not plot:
            return
        print('>> Plotting localization...')
        nodes = list(graph.nodes.values())
        probabilities = [node.probability for node in nodes]
        if len(graph.aggregated_p_kt) < 1:
            agg_probabilities = [0.0] * len(probabilities)
        else:
            agg_probabilities = graph.aggregated_p_kt

        images = []
        result = self.get_result(image_number, loop_id)

        for i, image in enumerate(images_original):
            alpha = agg_probabilities[i]
            gamma = min(agg_probabilities[i] + 0.2, 1.0)
            blank = np.zeros_like(image)
            if i != loop_id:
                jet_color = cv2.applyColorMap(np.array([[[alpha*255]]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                color_rect = (int(jet_color[0]), int(jet_color[1]), int(jet_color[2]))
                image = cv2.rectangle(image.copy(), (0, 0), (self.width-1, self.height-1), color_rect, 20)
                blended = cv2.addWeighted(image, gamma, blank, 1-gamma, 0)
                images.append(blended)
            else:
                color_rect = (int(254), int(254), int(254))
                image = cv2.rectangle(image.copy(), (0, 0), (self.width-1, self.height-1), (245,71,201), 20)
                cv2.putText(image, 'LOCALIZED', (60, 170), font, fontScale, color_rect, lineType)
                images.append(image)

        n_rows = self.rows
        rows = []
        for n in range(0, n_rows):
            if self.cols*(n+1) > len(images):
                end = len(images)
                img_h = cv2.hconcat(images[self.cols*n:end])
                if len(rows) < 1:
                    rows.append(img_h)
                else:
                    height, width, channels = rows[0].shape
                    mod_img = np.zeros((height, width, 3), np.uint8)
                    border_width = (width - img_h.shape[1]) // 2
                    mod_img = cv2.copyMakeBorder(img_h, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    rows.append(mod_img)
            else:
                end = self.cols*(n+1)
                img_h = cv2.hconcat(images[self.cols*n:end])
                rows.append(img_h)

        img_final = cv2.vconcat(rows)
        
        # Create GUI shape around map image
        height, width, channels = img_final.shape
        top = 50
        if len(self.reject_images) > 0:
            bottom = self.height + 50
        else:
            bottom = 50
        left = 30
        right = 475
        img_final = cv2.copyMakeBorder(img_final, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(40, 40, 40)) #map
        cv2.putText(img_final, 'WITHDRAWAL MAP SEQ_027', (int(width/2) - 80, 35), font, fontScale, (10, 255, 10), lineType)

        # Plot current frame
        frame = self.gui_plot_current_frame(current_image, state)
        frame = cv2.resize(frame, (450, 375))
        frame_height, frame_width, frame_channels = frame.shape
        img_final[500:500+frame_height, 40+width:40+width+frame_width, :] = frame # current frame
        cv2.putText(img_final, 'FRAME SEQ_035', (width+int(frame_width/2) - 80, 480), font, fontScale, (10, 255, 10), lineType)

        # Plot top-5 reject if exists
        if len(self.reject_images) > 0:
            indices = np.argsort(-graph.reject_scores, axis=0)[:13]
            reject_images = []
            for i in indices:
                reject_images.append(self.reject_images[i])

            reject = cv2.hconcat(reject_images)
            blank = np.zeros_like(reject)
            alpha = 0.6 if graph.is_reject else 0.2
            jet_color = cv2.applyColorMap(np.array([[[alpha*255]]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            color_rect = (int(jet_color[0]), int(jet_color[1]), int(jet_color[2]))
            reject = cv2.rectangle(reject.copy(), (0, 0), (reject.shape[1]-1, reject.shape[0]-1), color_rect, 20)
            blended = cv2.addWeighted(reject, alpha, blank, 1-alpha, 0)
            
            reject_height, reject_width, reject_channels = blended.shape
            img_final[90+height:90+height+reject_height, 30:30+reject_width, :] = blended # reject
            cv2.putText(img_final, 'REJECT NODE', (int(width/2) - 50, 80+height), font, fontScale, (10, 255, 10), lineType)

        colon_clone_img, point = self.plot_colon_trajectory(graph, self.colon_img)
        colon_plot = self.plot_colon(graph, loop_id, colon_clone_img, point)
        colon_height, colon_width, colon_channels = colon_plot.shape

        img_final[1000:1000+colon_height, 40+width:40+width+colon_width, :] = colon_plot # localization
        cv2.putText(img_final, 'LOCALIZATION', (40+width+int(colon_width/2) - 110, 980), font, fontScale, (10, 255, 10), lineType)

        # Plot result
        if result == 'CORRECT':
            cv2.putText(img_final, result, (40+width+int(colon_width/2) - 130, 1600), font, 2, (10, 255, 10), 3, lineType)
        elif result == 'WRONG':
            cv2.putText(img_final, result, (40+width+int(colon_width/2) - 90, 1600), font, 2, (10, 10, 255), 3, lineType)
        elif result == 'NONE':
            cv2.putText(img_final, result, (40+width+int(colon_width/2) - 90, 1600), font, 2, (211, 211, 211), 3, lineType)
        else:
            cv2.putText(img_final, result, (40+width+int(colon_width/2) - 238, 1600), font, 2, (0, 215, 255), 3, lineType)
        
        frame_id = graph.saved_frames
        graph.saved_frames += 1
        self.append_previous_frames(frame)

        filename = join(self.save_localization_path, str(frame_id).zfill(5) + '.png')
        cv2.imwrite(filename, cv2.resize(img_final, (1875, 800) ))

        # cv2.imshow('final', cv2.resize(img_final, (1875, 800) ))
        # cv2.waitKey(wait)

    def gui_plot_current_frame(self, image, state, plot = True):
        if not plot:
            return
        state = state.split('-')[0]
        print(f'>> Plotting current frame... {state}')
        current_image = cv2.imread(image)
        current_image = cv2.resize(current_image, self.frame_size)
        cv2.putText(current_image, state, (20, 40), font, fontScale, (10, 255, 10), lineType)
        return current_image
    
    def plot_proto_node_vertical(self, node):
        current_node = []
        if node is not None:
            for i, frame in enumerate(node.image_paths):
                current_image = cv2.imread(frame)
                current_image = cv2.resize(current_image, self.frame_size)
                current_node.append(current_image)

        while len(current_node) < 4:
            current_node.append(self.default_image)

        img_final = cv2.vconcat(current_node)
        return img_final
    
    def plot_previous_frames(self):
        previous = []
        for i, frame in enumerate(self.previous_frames):
            previous.append(frame)

        # Reverse list
        previous = previous[::-1]

        while len(previous) < 4:
            previous.append(self.default_image)

        img_final = cv2.vconcat(previous)
        return img_final
    
    def plot_colon_trajectory(self, graph, colon_img):
        colon_clone = colon_img.copy()
        nodes = list(graph.nodes.values())
        probabilities = [node.probability for node in nodes]

        max_point = 0
        max_prob = 0

        if len(graph.aggregated_p_kt) < 1:
            agg_probabilities = [0.0] * len(probabilities)
        else:
            agg_probabilities = graph.aggregated_p_kt
        #print(probabilities)
        state = 0
        counter = 0
        last_pose = 0
        nodes_added_in_region = 0
        for n_point, point in enumerate(positions):
            if counter >= regions[state]:
                counter = 0
                state += 1
                nodes_added_in_region = 0
            nodes_per_point = math.ceil(mapping[state]/regions[state])
            if last_pose+nodes_per_point <= len(probabilities):
                end = last_pose+nodes_per_point
            else:
                end = len(probabilities)

            nodes_added_in_region += nodes_per_point
            # If we get to next section, reduce last points coverage
            if nodes_added_in_region > mapping[state]:
                end = end - (nodes_added_in_region - mapping[state])

            

            probability = np.amax(agg_probabilities[last_pose:end])
            if probability > max_prob:
                max_prob = probability
                max_point = point
            # print(probabilities[last_pose:end])
            # print('PROB', probability)
            jet_color = cv2.applyColorMap(np.array([[[probability*255]]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            color = (int(jet_color[0]), int(jet_color[1]), int(jet_color[2]))

            cv2.circle(colon_clone, point, 5, color, -1)

            last_pose = end
            counter += 1

        return colon_clone, max_point
    
    def plot_colon(self, graph, loop_id, colon_trajectory, point):

        if loop_id != -1:
            circle_coords = point
            output = colon_trajectory
            radius = 30
            cv2.circle(output, circle_coords, radius, (245,71,201), 5)
        else:
            output = colon_trajectory

        return cv2.resize(output, (450, 450))