import argparse
from models.models import model_dict

def get_args():
    parser = argparse.ArgumentParser(
        description="Distill knowledge from a teacher model into a student model"
    )

    parser.add_argument("--teacher_model", type=str, required=True,
        choices=list(model_dict.keys()), help="Teacher model architecture")

    parser.add_argument("--student_model", type=str, required=True,
        choices=list(model_dict.keys()), help="Student model architecture")

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

    parser.add_argument("--teacher_batch_size", type=int, default=128,
        help="Teacher training batch size (default: 128)")

    parser.add_argument("--teacher_lr", type=float, default=0.001,
        help="Teacher learning rate (default: 0.001)")

    parser.add_argument("--teacher_num_epochs", type=int, default=10,
        help="Number of epochs to train the teacher (default: 10)")

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

    return parser.parse_args()
