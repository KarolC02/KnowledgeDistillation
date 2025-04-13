import argparse
from models.models import model_dict

def get_args():
    parser = argparse.ArgumentParser(
        description="Distill knowledge from a teacher model into a student model"
    )

    parser.add_argument(
        "--teacher_model", 
        type=str, 
        required=True,
        choices=list(model_dict.keys()),
        help=f"Teacher model architecture. Choices: {list(model_dict.keys())}"
    )

    parser.add_argument(
        "--student_model", 
        type=str, 
        required=True,
        choices=list(model_dict.keys()),
        help=f"Student model architecture. Choices: {list(model_dict.keys())}"
    )

    parser.add_argument(
        "--temperature", 
        type=float, 
        default=4.0,
        help="Distillation temperature (default: 4.0)"
    )

    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="Weight between soft loss and hard loss (default: 0.5)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for distillation (default: 128)"
    )

    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate for student (default: 0.001)"
    )

    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=10,
        help="Number of epochs to distill (default: 10)"
    )

    parser.add_argument(
        "--teacher_logits_path",
        type=str,
        default="teacher_logits.pt",
        help="Path to saved teacher logits (default: teacher_logits.pt)"
    )

    parser.add_argument(
        "--logdir",
        type=str,
        default="logs/distillation",
        help="Directory to save TensorBoard logs (default: logs/distillation)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use DataParallel for student model"
    )

    return parser.parse_args()
