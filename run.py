import argparse
import torch
import os
from experiment import Exp_Classification
from utils import set_seed


def main():
    # Set random seed for reproducibility
    set_seed(42)

    parser = argparse.ArgumentParser(description='Multimodal HRM Species Classification')

    # ----- Model -----
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_class', type=int, default=65)
    parser.add_argument('--enc_in', type=int, default=1, help='Number of input variables')
    parser.add_argument('--use_text', action='store_true', default=True, help='Use text modality with BERT')
    parser.add_argument('--use_gate', action='store_true', default=False, help='Use head-specific gate (Qwen style)')
    parser.add_argument('--no_gate', dest='use_gate', action='store_false', help='Disable head-specific gate')

    # ----- Optimization -----
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_epochs', type=int, default=3000)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--seq_len', type=int, default=3478)
    parser.add_argument('--dropout', type=float, default=0.1)

    # ----- Contrastive Loss -----
    parser.add_argument('--contrastive_weight', type=float, default=0, help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')

    # ----- System -----
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--use_gpu', action='store_true', default=True)

    args = parser.parse_args()

    # Decide device
    if args.use_gpu and torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        print("WARNING: Using CPU, slow!")

    # Ensure checkpoint directory
    os.makedirs(args.checkpoints, exist_ok=True)

    print("Training Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")


    # ----- Run Experiment -----
    print("\nInitializing experiment...")
    exp = Exp_Classification(args)

    print("\nStarting training...")
    exp.train()


if __name__ == '__main__':
    main()
