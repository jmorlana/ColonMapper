"""
Evaluation functions for the localization results.
First, it reads the ground truth labels from a text file.
Second, it reads the node labels from a text file.
Then, it reads the localization results from a text file.
Finally, it computes the precision and recall for the localization results.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from settings import EVAL_PATH, DATA_PATH

from os.path import join

def parse_range(range_str):
    x_str, y_str = range_str.split('-')
    x, y = int(x_str), int(y_str)

    return list(range(x, y + 1))

def read_grund_truth_labels(labels_file):
    """Read the labels from a text file.
    Labels are in the format:
    image_number: label
    Some images have no correspondences, in this case the node number is 'N - previous_label'.
    """

    ground_truth = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_number, label = line.split(":")
                image_number = int(image_number.strip())
                label = label.strip()
                ground_truth[image_number] = label

                

    return ground_truth

def read_localizations(localizations_file):
    """Read the localizations from a text file.
    Localizations are in the format:
    image_number: node_number - score
    Some images have no correspondences, in this case the node number is 'N'.
    """

    localizations = {}
    scores = {}
    with open(localizations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                image_number, node_number = line.split(":")
                image_number = int(image_number.strip())
                node_number = node_number.strip()

                # Check if the node has a score
                node_number = node_number.split("-")
                if len(node_number) == 2:
                    node_number, score = node_number
                    node_number = node_number.strip()
                    score = float(score.strip())
                    scores[image_number] = score
                elif len(node_number) == 3:
                    node_number, score, real_node = node_number
                    node_number = node_number.strip()
                    score = float(score.strip())
                    scores[image_number] = score
                else:
                    node_number = node_number[0].strip()

                if node_number == 'N':
                    localizations[image_number] = node_number
                else:
                    localizations[image_number] = int(node_number)

    if len(scores) == 0:
        scores = None

    return localizations, scores

def read_node_labels(node_labels_file):
    """Read the node labels from a text file.
    Node labels are in the format:
    Label: ranges of nodes in the format: start_node_0-end_node_0, start_node_1-end_node_1, ...
    """
    
    node_labels = {}
    with open(node_labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label, intervals = line.split(":")
                label = label.strip()
                if len(intervals) > 0:
                    intervals = intervals.strip()
                    intervals = intervals.split(",")
                    for interval in intervals:
                        nodes = parse_range(interval)
                        for node in nodes:
                            node_labels[node] = label
                               
    return node_labels

def update_localizations(localizations, node_labels, scores=None, threshold=0.5):
    """Update the localizations with the node labels.
    """

    new_localizations = {}
    for image_number, node_number in localizations.items():
        # If score exists and below threshold, localization is not accepted
        if scores is not None:
            score = scores[image_number]
            if score < threshold:
                new_localizations[image_number] = 'N'
                continue

        # Otherwise, update localization with the node label
        if node_number == 'N':
            new_localizations[image_number] = 'N'
        else:
            new_localizations[image_number] = node_labels[node_number]

    return new_localizations

def compute_precision_recall(ground_truth, localizations):
    """Compute the precision and recall for the localization results.
    """

    # Compute the number of correctly found nodes with and without correspondences
    true_positives = 0
    true_positives_no_correspondences = 0
    total_positives = 0
    total_positives_no_correspondences = 0
    skipped = 0
    for image_number, est_label in localizations.items():
        if image_number in ground_truth:
            gt_label = ground_truth[image_number]
            if gt_label.startswith('N'):
                if est_label == 'N':
                    true_positives_no_correspondences += 1
                skipped += 1

            else:
                if est_label == gt_label:
                    true_positives += 1

            # Count total positives
            if est_label == 'N':
                total_positives_no_correspondences += 1
            else:
                total_positives += 1
            
    relevant = len(ground_truth) - skipped # total colon regions 
    relevant_no_correspondences = skipped # total N
            

    # Compute the precision and recall
    # precision = true_positives / relevant
    # precision_no_correspondences = true_positives_no_correspondences / relevant_no_correspondences
    # recall = true_positives / len(ground_truth)
    # recall_no_correspondences = true_positives_no_correspondences / len(ground_truth)

    precision = true_positives / total_positives
    precision_no_correspondences = true_positives_no_correspondences / total_positives_no_correspondences
    recall = true_positives / relevant
    recall_no_correspondences = true_positives_no_correspondences / relevant_no_correspondences
    
    # print(f"Total positives / true positives / relevant elements: {total_positives, true_positives, relevant}")
    # print(f"Total positives (N) / true positives (N) / relevant elements (N): {total_positives_no_correspondences, true_positives_no_correspondences, relevant_no_correspondences}")

    return precision, true_positives, recall

def compute_precision_recall_no_N(ground_truth, localizations):
    """Compute the precision and recall for the localization results.
    """

    # Compute the number of correctly found nodes with and without correspondences
    true_positives = 0
    true_positives_no_correspondences = 0
    total_positives = 0
    total_positives_no_correspondences = 0
    skipped = 0
    for image_number, est_label in localizations.items():
        if image_number in ground_truth:
            gt_label = ground_truth[image_number]
            if gt_label.startswith('N'):
                skipped += 1
                continue

            else:
                if est_label == gt_label:
                    true_positives += 1

            # Count total positives
            if est_label == 'N':
                total_positives_no_correspondences += 1
            else:
                total_positives += 1
            
    relevant = len(ground_truth) - skipped # total colon regions 
    relevant_no_correspondences = skipped # total N
            
    # Compute the precision and recall
    if total_positives == 0:
        precision = 1.0
    else:
        precision = true_positives / total_positives
    precision_no_correspondences = 0
    recall = true_positives / relevant
    recall_no_correspondences = 0
    
    # print(f"Total positives / true positives / relevant elements: {total_positives, true_positives, relevant}")
    # print(f"Total positives (N) / true positives (N) / relevant elements (N): {total_positives_no_correspondences, true_positives_no_correspondences, relevant_no_correspondences}")

    return precision, true_positives, recall

def PR_curve_threshold(models, results):
    """Plot the results in a Precision-Recall curve for every dataset.
    """

    datasets = ['027', 'entry_cross', 'cross']
    #models = ['resnet101conv5_gem_midl', 'resnet50conv4_netvlad_0_0_640_hard_resize_layer2']
    categories = ['single_image', 'single_image_reject', 'bayesian', 'bayesian_reject']

    dataset_title = {'027': 'withdrawal_027 vs entry_027', 'entry_cross': 'entry_035 vs withdrawal_027', 'cross': 'withdrawal_035 vs withdrawal_027'}

    cat_names = {'single_image': 'SI', 'single_image_reject': 'SI + R', 
                 'bayesian': 'Bayes', 'bayesian_reject': 'Bayes + R'}

    colors = {'resnet101conv5_gem_midl': 'red', 
              'resnet50conv4_gem_0_0_640_hard_resize_layer2': 'orange', 
              'resnet50conv4_netvlad_0_0_640_hard_resize_layer2': 'blue'}
    
    linestyles = {'single_image': 'dotted',
                'single_image_reject': 'solid',
                'bayesian': 'dotted',
                'bayesian_reject': 'solid',
                'bayesian_reject_alpha': (5, (10, 3))} # loosely dotted
    
    names = {'resnet101conv5_gem_midl': 'Morlana21', 
              'resnet50conv4_gem_0_0_640_hard_resize_layer2': 'ours (GeM)', 
              'resnet50conv4_netvlad_0_0_640_hard_resize_layer2': 'ours (NV)'}
    
    op_points = {'resnet101conv5_gem_midl': {'single_image': 0.85,
                                             'single_image_reject': 0.85},
                 'resnet50conv4_gem_0_0_640_hard_resize_layer2': {'single_image': 0.8, 
                                                                  'single_image_reject': 0.8,
                                                                  'bayesian': 0.5,
                                                                  'bayesian_reject': 0.5},
                'resnet50conv4_netvlad_0_0_640_hard_resize_layer2': {'single_image': 0.55, 
                                                                    'single_image_reject': 0.55,
                                                                    'bayesian': 0.5,
                                                                    'bayesian_reject': 0.5}}

    # Create matplotlib figure with 3 subplots, each axs correspond to one dataset
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the PR curve for every dataset
    for i, dataset in enumerate(datasets):
        axs[i].set_aspect('equal')
        axs[i].tick_params(axis='both', which='major', labelsize=12)
        axs[i].set_title(f'{dataset_title[dataset]}', fontsize=18)
        axs[i].set_xlabel('Recall', fontsize=18)
        axs[i].set_ylabel('Precision', fontsize=18)
        axs[i].set_xlim([0, 1])
        axs[i].set_ylim([0, 1])
        axs[i].grid(True)

        for model in models:
            for category in categories:
                key = f'{model}_{category}'
                if category.startswith('bayesian') and model == 'resnet101conv5_gem_midl':
                    continue
                if category.startswith('single_image') and model == 'resnet50conv4_gem_0_0_640_hard_resize_layer2':
                    continue
                precision = []
                recall = []
                for threshold in results[key][dataset].keys():
                    precision.append(results[key][dataset][threshold]['precision'])
                    recall.append(results[key][dataset][threshold]['recall'])

                if 'bayesian' in category and model == 'resnet50conv4_netvlad_0_0_640_hard_resize_layer2':
                    color = 'green'
                else:
                    color = colors[model]
                
                linestyle = linestyles[category]
                name = names[model]

                axs[i].plot(recall, precision, label=f'{name} - {cat_names[category]}', color=color, linestyle=linestyle)

                operating_point = op_points[model][category]
                marker_x = results[key][dataset][operating_point]['recall']
                marker_y = results[key][dataset][operating_point]['precision']
                axs[i].scatter(marker_x, marker_y, color=color, marker='*', s=150)

                # Print operating point for every dataset and model
                print(f'{dataset} - {model} - {category}: {operating_point} - {results[key][dataset][operating_point]}')

        axs[i].legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('endomapper_sequences.pdf', format='pdf', dpi=300, bbox_inches='tight')  # Adjust the file name and dpi as needed
    plt.show()


def custom_sort(category):
    if category.startswith('single_'):
        return (0, category)
    elif category.startswith('bayesian'):
        return (1, category)
    else:
        return (2, category)
    
def get_categories(default_model):
    """List all the dirs inside the default model folder."""
    
    categories = []
    for root, dirs, files in os.walk(default_model):
        for dir in dirs:
            if dir.startswith('single_') or dir.startswith('bayesian'):
                categories.append(dir)

    sorted_categories = sorted(categories, key=custom_sort)

    return sorted_categories


def get_metrics(models):
    datasets = ['027', 'entry_cross', 'cross']
    categories = ['single_image', 'single_image_reject', 'bayesian', 'bayesian_reject']

    # Threshold from 0.01 to 1.0, step 0.01
    thresholds = list(np.round(np.arange(0.01, 1.0, 0.01), decimals=2))
    logs_path = EVAL_PATH
    labels_path = DATA_PATH / 'endomapper'

    results = {}
    for category in categories:
        for model in models:
            key = f'{model}_{category}'
            results[key] = {}

            for dataset in datasets:
                gt_file = f'eval_{dataset}.txt'
                node_labels_file = f'labels_{dataset}.txt'
                localizations_file = f'{dataset}_localizations.txt'

                ground_truth_path = join(labels_path, gt_file)
                node_labels_path = join(labels_path, 'canonical_maps', node_labels_file)
                localizations_path = join(logs_path, model, category, dataset, localizations_file)

                # Check if localization file exists
                if not os.path.exists(localizations_path):
                    print(f'File {localizations_path} does not exist')
                    results[key][dataset] = {'precision': 0, 'true_positives': 0, 'recall': 0}
                    continue

                ground_truth = read_grund_truth_labels(ground_truth_path)
                node_labels = read_node_labels(node_labels_path)
                localizations, scores = read_localizations(localizations_path)

                results[key][dataset] = {}
                for threshold in thresholds:
                    # Update localizations with the node labels
                    new_localizations = update_localizations(localizations, node_labels, scores, threshold=threshold)

                    # if 'reject' in category and 'bayesian' not in category:
                    #     precision, true_positives, recall = compute_precision_recall(ground_truth, localizations)
                    # else:
                    #     precision, true_positives, recall = compute_precision_recall_no_N(ground_truth, localizations)

                    precision, true_positives, recall = compute_precision_recall_no_N(ground_truth, new_localizations)

                    results[key][dataset][threshold] = {'precision': precision, 'true_positives': true_positives, 'recall': recall}

    return results, categories

if __name__ == '__main__':
    models = ['resnet101conv5_gem_midl', 'resnet50conv4_gem_0_0_640_hard_resize_layer2', 'resnet50conv4_netvlad_0_0_640_hard_resize_layer2']
    results, categories = get_metrics(models)
    PR_curve_threshold(models, results)