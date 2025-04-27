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
        default="results",
        help="Directory to save TensorBoard logs"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run parallel on multiple GPUs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers for dataloaders (default: 16)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny-imagenet",
        help="Dataset to train on a dataset"
    )

    parser.add_argument(
        "--save_checkpoint_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training (default: None)"
    )

    parser.add_argument(
        "--adapt_model",
        action="store_true",
        help="Adapt model to num of classses of the dataset (1000 -> 200 in case of tiny image-met)"
    )


    return parser.parse_args()
