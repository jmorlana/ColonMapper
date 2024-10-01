import cv2
import numpy as np
import math
import os
from os.path import join
from settings import root
import matplotlib.pyplot as plt
import shutil

from utils.point_matrix import positions, mapping
from settings import EVAL_PATH

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20, 20)
fontScale = 1
lineType = 2

# section ids -- not accurate
cecum_id = 4280
ascending_id = 5310
transverse_id = 8990
descending_id = 9500
sigmoid_id = 10190
rectum_id = 12500
retroflexion_id = 12900

# circle coords
cecum = (40, 160)
ascendent = (40, 110)
transverse = (130, 45)
descendent = (210, 115)
sigmoid = (170, 175)
rectum = (130, 180)
retro = (130, 230) 

colon_img = cv2.imread(str(root / 'assets/colon_background.png'))
default_image = cv2.resize(cv2.imread(str(root / 'assets/default.jpg')), (300, 250))
default_image_mapping = cv2.resize(cv2.imread(str(root / 'assets/default.jpg')), (250, 200))

# Save path for mapping frames
# path = '/home/jmorlana/Experiments/ColonMapper_results/mapping_frames/'
save_mapping_path = EVAL_PATH / 'mapping_frames'
if not save_mapping_path.exists():
    os.makedirs(save_mapping_path)
else:
    # Clear all images in the folder
    shutil.rmtree(save_mapping_path)
    os.makedirs(save_mapping_path)
save_mapping_path = str(save_mapping_path)

# Visualization functions for mapping
def plot_proto_node_vertical_mapping(node):
    current_node = []
    if node is not None:
        image_paths = node.image_paths

        for i, frame in enumerate(image_paths):
            current_image = cv2.imread(frame)
            current_image = cv2.resize(current_image, (300, 250))
            cv2.putText(current_image, str(i+1), (20, 40), font, fontScale, (10, 255, 10), lineType)
            current_node.append(current_image)

    while len(current_node) < 4:
        current_node.append(default_image)

    node_list = [current_node[0], current_node[1], current_node[2], current_node[-1]]

    img_final = cv2.vconcat(node_list)
    return img_final

def gui_plot_current_frame(image, state, plot = True):
    if not plot:
        return
    current_image = cv2.imread(image)
    current_image = cv2.resize(current_image, (300, 250))
    cv2.putText(current_image, state, (20, 40), font, fontScale, (10, 255, 10), lineType)
    return current_image

def plot_colon_trajectory_mapping(graph, colon_img):
    colon_clone = colon_img.copy()
    nodes = list(graph.nodes.values())
   
    # Estimate how many points to plot
    n_nodes = len(nodes)
    #print(probabilities)
    points_to_plot = 0
    break_loop = False

    for state, map_nodes in enumerate(mapping):
        if n_nodes >= map_nodes:
            n_nodes -= map_nodes
            nodes_to_sum = map_nodes
        else:
            nodes_to_sum = n_nodes
            break_loop = True
        
        points_to_plot += nodes_to_sum
        # if state != 4:
        #     points_to_plot += nodes_to_sum
        # else:
        #     points_to_plot += int(nodes_to_sum/4)

        if break_loop:
            break

    points = positions[0:points_to_plot]


    for point in points:
        cv2.circle(colon_clone, point, 5, (255,0,0), -1)
    
    if len(points) > 0:
        cv2.circle(colon_clone, points[-1], 5, (0,0,255), -1) # last is plotted in red

    return cv2.resize(colon_clone, (450, 450))

def gui_plot_mapping(graph, current_image, state, current_node, plot=False, wait = 100):
    # GUI is hardcoded for mapping
    if not plot:
        return
    nodes = list(graph.nodes.values())

    images = []

    for n, node in enumerate(nodes):
        img = node.image.copy()
        #img = cv2.resize(img, (300, 250))
        node_text = str(node.id) + ' - ' + str(len(node.image_paths))
        cv2.putText(img, node_text, (20, 50), font, fontScale, (10, 255, 10), lineType)
        cv2.putText(img, str(node.n_frame), (20, 80), font, 1, (10, 255, 10), 1)
        images.append(img)

    while len(images) < 120:
        images.append(default_image_mapping)

    n_rows = math.ceil(len(images)/13)
    rows = []
    for n in range(0, n_rows):
        if 13*(n+1) > len(images):
            end = len(images)
            img_h = cv2.hconcat(images[13*n:end])
            if len(rows) < 1:
                rows.append(img_h)
            else:
                height, width, channels = rows[0].shape
                mod_img = np.zeros((height, width, 3), np.uint8)
                border_width = (width - img_h.shape[1]) // 2
                mod_img = cv2.copyMakeBorder(img_h, 0, 0, border_width, border_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                rows.append(mod_img)
        else:
            end = 13*(n+1)
            img_h = cv2.hconcat(images[13*n:end])
            rows.append(img_h)

    img_final = cv2.vconcat(rows)

    # Create GUI shape
    height, width, channels = img_final.shape
    img_final = cv2.copyMakeBorder(img_final, 50, 30, 30, 470, cv2.BORDER_CONSTANT, value=(40, 40, 40)) #map
    cv2.putText(img_final, 'MAPPING WITHDRAWAL SEQ_027', (int(width/2) - 80, 35), font, fontScale, (10, 255, 10), lineType)

    frame = gui_plot_current_frame(current_image, state)
    frame_height, frame_width, frame_channels = frame.shape
    img_final[80:80+frame_height, 100+width:100+width+frame_width, :] = frame # current frame
    cv2.putText(img_final, 'FRAME', (100+width+int(frame_width/2) - 40, 60), font, fontScale, (10, 255, 10), lineType)

    node = plot_proto_node_vertical_mapping(current_node)
    node_height, node_width, node_channels = node.shape
    img_final[430:430+node_height, 100+width:100+width+node_width, :] = node # current node
    if current_node is not None:
        node_text = 'NODE - ' + str(len(current_node.image_paths)) + ' FRAMES'
    else:
        node_text = 'NODE - 0 FRAMES'
    cv2.putText(img_final, node_text, (width+int(frame_width/2) - 50, 410), font, fontScale, (10, 255, 10), lineType)

    colon_plot = plot_colon_trajectory_mapping(graph, colon_img)
    colon_height, colon_width, colon_channels = colon_plot.shape

    img_final[1600:1600+colon_height, 40+width:40+width+colon_width, :] = colon_plot # localization
    cv2.putText(img_final, 'MAPPED COLON', (40+width+int(colon_width/2) - 140, 1580), font, fontScale, (10, 255, 10), lineType)

    frame_id = graph.saved_frames
    graph.saved_frames += 1

    filename = join(save_mapping_path, str(frame_id).zfill(5) + '.png')
    cv2.imwrite(filename, cv2.resize(img_final, (1875, 800) ))

    # cv2.imshow('final', cv2.resize(img_final, (1875, 800) ))
    # cv2.waitKey(wait)

def plot_probabilities(graph):
    # Bar plots
    if graph.first_plot == False:
        plt.clf()
        #plt.cla()
    else:
        graph.first_plot = False
    fig, (pred, score, update) = plt.subplots(3, 1, figsize=(15, 15), num = 'Bayesian localization')

    nodes = list(graph.nodes.values())

    predicted = list(graph.p_kt_pred)
    scores = list(graph.scores)
    nodes_id = [node.id for node in nodes]
    probabilities = [node.probability for node in nodes]
    
    x_min = 0
    x_max = len(nodes_id) + 1

    pred.bar(nodes_id, predicted, width = 0.2, align='edge')
    pred.set_xticks(nodes_id) 
    pred.set_xticklabels(nodes_id)
    pred.set_ylim(0, 1)
    pred.set_xlim(x_min, x_max)
    pred.set_title('Prior')

    score.bar(nodes_id, scores, width = 0.2, align='edge')
    score.set_xticks(nodes_id) 
    score.set_xticklabels(nodes_id)
    score.set_ylim(0, 1)
    score.set_xlim(x_min, x_max)
    score.set_title('Likelihood')

    update.bar(nodes_id, probabilities, width = 0.2, align='edge')
    update.set_xticks(nodes_id) 
    update.set_xticklabels(nodes_id)
    update.set_ylim(0, 1)
    update.set_xlim(x_min, x_max)
    update.set_title('Posterior')
    
    plt.pause(0.0000001)
    plt.tight_layout()
    plt.plot()

def plot_LoFTR_matches(image0, image1, kpts0, kpts1, mkpts0,
                        mkpts1, color, text, path=None,
                        show_keypoints=False, margin=10,
                        opencv_display=False, opencv_title='',
                        small_text=[]):

    H0, W0, C0 = image0.shape
    H1, W1, C1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 255, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        out = cv2.resize(out, (int(out.shape[1]), int(out.shape[0])))
        cv2.imshow(opencv_title, out)
        #cv2.waitKey(0)

    return out
