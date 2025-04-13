import argparse
from models.models import model_dict 

def get_args():
    parser = argparse.ArgumentParser(
        description="Train a model on Tiny ImageNet"
    )

    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(model_dict.keys()),
        help=f"Model architecture to use. Choices: {list(model_dict.keys())}"
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (default: 10)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/tiny_image_net",
        help="Directory to save TensorBoard logs"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run parallel on multiple GPUs"
    )

    return parser.parse_args()
