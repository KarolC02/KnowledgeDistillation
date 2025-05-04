import argparse
from models.models import model_dict

def get_args():
    parser = argparse.ArgumentParser(
        description="Distill knowledge from a teacher model into a student model"
    )

    parser.add_argument("--student_model", type=str, required=True,
        choices=list(model_dict.keys()), help="Student model architecture")

    parser.add_argument(
        "--teacher_checkpoint_path",
        type=str,
        required=True,
        help="Path to a pretrained teacher checkpoint (.pth). Overrides automatic path construction."
    )
    parser.add_argument("--temperature", type=float, default=4.0,
        help="Distillation temperature (default: 4.0)")

    parser.add_argument("--alpha", type=float, default=0.5,
        help="Weight between soft loss and hard loss (default: 0.5)")

    parser.add_argument("--batch_size", type=int, default=128,
        help="Student batch size (default: 128)")

    parser.add_argument("--lr", type=float, default=0.001,
        help="Student learning rate (default: 0.001)")

    parser.add_argument("--num_epochs", type=int, default=10,
        help="Number of distillation epochs (default: 10)")

    parser.add_argument("--dataset", type=str, default="tiny-imagenet",
        help="Dataset name (default: tiny-imagenet)")

    parser.add_argument("--logdir", type=str, default="results",
        help="Base directory for all logs and experiments")

    parser.add_argument("--logits_dir", type=str, default="saved_logits",
        help="Where to cache teacher logits")

    parser.add_argument("--modeldir", type=str, default="saved_models",
        help="Where to save distilled student models")

    parser.add_argument("--parallel", action="store_true",
        help="Use DataParallel for teacher and student")
    
    parser.add_argument("--adapt_model", action="store_true",
        help="Adapt model output layer to match number of classes (e.g., 1000 -> 200)")
    
    parser.add_argument("--num_workers", type=int, default=16, 
        help="Number of workers for data loading")
    
    parser.add_argument("--num_classes", type=int, default=200,
        help="How many classification classes in the dataset (default : 200, for tiny-image-net)" )

    parser.add_argument("--optimizer", type=str, default="Adam",
        help="Model optimiizer (default: adam)" )
    
    parser.add_argument("--logits_path", type=str, default="-",
        help="Model optimiizer (default: adam)" )

    parser.add_argument(
        "--lr_decay_every",
        type=int,
        default=0,
        help="Decay learning rate every N epochs (0 means no decay)"
    )

    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=0.1,
        help="Factor to decay learning rate by (e.g., 0.1 means divide by 10)"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization), default is 0.0"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability to apply in classifier head (default: 0.0)"
    )


    return parser.parse_args()
