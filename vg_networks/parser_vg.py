
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18conv4",
                        choices=["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                                 "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5",
                                 "cct384", "vit"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="netvlad", choices=["netvlad", "gem", "spoc", "mac", "rmac", "crn", "rrm",
                                                                               "cls", "seqpool"])
    parser.add_argument('--netvlad_clusters', type=int, default=64, help="Number of clusters for NetVLAD layer.")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    parser.add_argument('--pretrain', type=str, default="imagenet", choices=['imagenet', 'gldv2', 'places'],
                        help="Select the pretrained weights for the starting network")
    parser.add_argument("--off_the_shelf", type=str, default="imagenet", choices=["imagenet", "radenovic_sfm", "radenovic_gldv1", "naver"],
                        help="Off-the-shelf networks from popular GitHub repos. Only with ResNet-50/101 + GeM + FC 2048")
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 14)))
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(-1, 14)))
    parser.add_argument("--resnet_layer", type=str, default="layer3", choices=['layer1', 'layer2', 'layer3'])
    parser.add_argument("--aspect_ratio", type=str, default="resize", choices=['central_crop', 'resize'])
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[480, 640], nargs=2, help="Resizing shape for images (HxW).")
    # Paths parameters
    parser.add_argument("--datasets",  nargs='+', default=["027", "035","entry_cross", "cross"], help="Sequences to test on")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--model_path", type=str, default="/home/jmorlana/deep-visual-geo-localization-benchmark/trained_models/cct384_netvlad/best_model.pth",
                        help="Model path to test (saved in ./trained_models/)")
    # ColonMapper parameters
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment")
    parser.add_argument('--reject_outliers', dest='reject_outliers', default=False, action='store_true',
                        help='Wheter to use reject_node frames or not')
    parser.add_argument('--reject_strategy', type=str, default='diffusion',
                        help='How to use the reject frames for localization')
    parser.add_argument("--likelihood_method", type=str, default='topk',
                        help="Method for estimating likelihood")
    parser.add_argument('--plot', dest='plot', default=False, action='store_true',
                        help='Wheter to show or not the GUI')
    parser.add_argument('--bayesian', dest='bayesian', default=False, action='store_true',
                        help='Wheter to use bayesian localization or not')
    parser.add_argument("--threshold_probability", type=float, default=0.9,
                        help="For Bayesian localization, probability threshold to accept a localization")
    parser.add_argument("--threshold_likelihood", type=float, default=0.5,
                        help="Reduce likelihood if score is not above certain threshold")
    parser.add_argument('--reload_graph', dest='reload_graph', default=False, action='store_true',
                        help='Wheter to extract new descriptors for a previously built graph')
    args = parser.parse_args()
    
    if args.aggregation == "crn" and args.resume is None:
        raise ValueError("CRN must be resumed from a trained NetVLAD checkpoint, but you set resume=None.")
    
    if torch.cuda.device_count() >= 2 and args.criterion in ['sare_joint', "sare_ind"]:
        raise NotImplementedError("SARE losses are not implemented for multiple GPUs, " +
                                  f"but you're using {torch.cuda.device_count()} GPUs and {args.criterion} loss.")
    
    if args.off_the_shelf in ["radenovic_sfm", "radenovic_gldv1", "naver"]:
        if args.backbone not in ["resnet50conv5", "resnet101conv5"] or args.aggregation != "gem" or args.fc_output_dim != 2048:
            raise ValueError("Off-the-shelf models are trained only with ResNet-50/101 + GeM + FC 2048")
    
    if args.backbone == "vit":
        if args.resize != [224, 224] and args.resize != [384, 384]:
            raise ValueError(f'Image size for ViT must be either 224 or 384 {args.resize}')
    if args.backbone == "cct384":
        if args.resize != [384, 384]:
            raise ValueError(f'Image size for CCT384 must be 384, but it is {args.resize}')
    
    if args.backbone in ["alexnet", "vgg16", "resnet18conv4", "resnet18conv5",
                         "resnet50conv4", "resnet50conv5", "resnet101conv4", "resnet101conv5"]:
        if args.aggregation in ["cls", "seqpool"]:
            raise ValueError(f"CNNs like {args.backbone} can't work with aggregation {args.aggregation}")
    if args.backbone in ["cct384"]:
        if args.aggregation in ["spoc", "mac", "rmac", "crn", "rrm"]:
            raise ValueError(f"CCT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls, seqpool]")
    if args.backbone == "vit":
        if args.aggregation not in ["cls", "gem", "netvlad"]:
            raise ValueError(f"ViT can't work with aggregation {args.aggregation}. Please use one among [netvlad, gem, cls]")

    return args
