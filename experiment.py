from dataset import TimeSeriesDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pickle as pkl
import pdb
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from pathlib import Path
import math
import json
from model import Model
from contrastive_loss import SupConLoss


warnings.filterwarnings('ignore')


class Exp_Classification(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        self.use_text = args.use_text if hasattr(args, 'use_text') else False
        self.model = Model(args, self.device, use_text=self.use_text).float().to(self.device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Trainable parameters:", trainable_params)
        print(f"Using text modality: {self.use_text}")



    def _get_data(self, flag):
        data_set = TimeSeriesDataset(flag, use_text=self.use_text)
        shuffle_flag = (flag == 'train')
        data_loader = DataLoader(
            data_set,
            batch_size=self.args.batch_size,
            shuffle=shuffle_flag,
            num_workers=0,
            drop_last=False
        )
        return data_set, data_loader

    def adjust_learning_rate(self, optimizer, epoch, args):
        # lr = args.learning_rate * (0.2 ** (epoch // 2))
        if args.lradj == 'type1':
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        elif args.lradj == 'type2':
            lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        elif args.lradj == "cosine":
            lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))

    def validation(self, vali_loader, criterion, return_details=False):
        trues, preds = [], []
        total_loss = 0.0
        all_probs = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                if self.use_text:
                    x_time, x_text, label = batch
                else:
                    x_time, label = batch

                # if len(label) <= 1:
                #     continue

                x_time = x_time.float().to(self.device)
                label = label.to(self.device)

                if self.use_text:
                    outputs, features = self.model(x_time, x_text)
                else:
                    outputs, features = self.model(x_time)

                # Calculate loss
                batch_y = label.long().view(-1)
                loss = criterion(outputs, batch_y).mean()
                total_loss += loss.item()

                preds.append(outputs.detach())
                trues.append(label)

        trues = torch.cat(trues, 0)
        trues = trues.flatten().cpu().numpy()

        preds = torch.cat(preds, dim=0)
        probs = F.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()

        avg_loss = total_loss / len(vali_loader)
        f1_micro = f1_score(trues, predictions, average='micro')
        f1_macro = f1_score(trues, predictions, average='macro')
        accuracy = np.mean(predictions == trues)
        self.model.train()

        if return_details:
            # Return detailed information for analysis
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'true_labels': trues,
                'predictions': predictions,
                'probabilities': probs.cpu().numpy()
            }
        else:
            return avg_loss, f1_micro, f1_macro, accuracy
    
    def train(self):
        train_dataset, train_loader = self._get_data('train')
        val_dataset, val_loader = self._get_data('val')
        test_dataset, test_loader = self._get_data('test')

        path = self.args.checkpoints
        if not os.path.exists(path):
            os.makedirs(path)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        contrastive_criterion = SupConLoss(temperature=self.args.temperature)
        best_val_loss = float('inf')
        best = None
        best_epoch = -1
        best_metrics = None
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                if self.use_text:
                    x_time, x_text, label = batch
                else:
                    x_time, label = batch

                if len(label) <= 1:
                    continue

                optimizer.zero_grad()

                x_time = x_time.float().to(self.device)
                label = label.to(self.device)

                if self.use_text:
                    output, features = self.model(x_time, x_text)
                else:
                    output, features = self.model(x_time)

                batch_y = label.long().view(-1)

                # CE Loss
                ce_loss = criterion(output, batch_y).mean()

                # Contrastive Loss
                con_loss = contrastive_criterion(features, batch_y)

                # Combined Loss
                loss = ce_loss + self.args.contrastive_weight * con_loss
                train_loss.append(loss.item())
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()



            train_loss = np.average(train_loss)
            val_loss, val_micro, val_macro, val_acc = self.validation(val_loader, criterion)
            test_loss, test_micro, test_macro, test_acc = self.validation(test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f}\n"
                "Vali-Loss: {3:.3f} Vali-Acc: {4:.3f} Vali-Micro: {5:.3f} Vali-Macro: {6:.3f}\n"
                "Test-Loss: {7:.3f} Test-Acc: {8:.3f} Test-Micro: {9:.3f} Test-Macro: {10:.3f}"
                .format(
                    epoch + 1,
                    len(train_loader),
                    train_loss,
                    val_loss, val_acc, val_micro, val_macro,
                    test_loss, test_acc, test_micro, test_macro
                )
            )
            if val_loss < best_val_loss:
                print("New Best Found!")
                best = (
                    "Best Model\n"
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f}\n"
                    "Vali-Loss: {3:.3f} Vali-Acc: {4:.3f} Vali-Micro: {5:.3f} Vali-Macro: {6:.3f}\n"
                    "Test-Loss: {7:.3f} Test-Acc: {8:.3f} Test-Micro: {9:.3f} Test-Macro: {10:.3f}"
                ).format(
                    epoch + 1,
                    len(train_loader),
                    train_loss,
                    val_loss, val_acc, val_micro, val_macro,
                    test_loss, test_acc, test_micro, test_macro
                )

                best_val_loss = val_loss
                best_epoch = epoch + 1

                # Get detailed metrics for the best model
                val_details = self.validation(val_loader, criterion, return_details=True)
                test_details = self.validation(test_loader, criterion, return_details=True)

                best_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss),
                    'validation': val_details,
                    'test': test_details
                }

                best_model_path = path + '/' + 'checkpoint_1.pth'
                torch.save(self.model.state_dict(), best_model_path)
            # if (epoch + 1) % 5 == 0:
            #     self.adjust_learning_rate(optimizer, epoch + 1, self.args)

        print(best)

        # Save best model metrics and misclassified samples
        if best_metrics is not None:
            self._save_best_model_analysis(best_metrics, path)

        return self.model

    def _save_best_model_analysis(self, metrics, save_path):
        """Save detailed analysis of the best model"""

        # Extract validation and test results
        val_true = metrics['validation']['true_labels']
        val_pred = metrics['validation']['predictions']
        val_probs = metrics['validation']['probabilities']

        test_true = metrics['test']['true_labels']
        test_pred = metrics['test']['predictions']
        test_probs = metrics['test']['probabilities']

        # Find misclassified samples
        val_misclassified_idx = np.where(val_true != val_pred)[0]
        test_misclassified_idx = np.where(test_true != test_pred)[0]

        # Create detailed misclassification info
        val_misclassified = []
        for idx in val_misclassified_idx:
            val_misclassified.append({
                'sample_index': int(idx),
                'true_label': int(val_true[idx]),
                'predicted_label': int(val_pred[idx]),
                'confidence': float(val_probs[idx, val_pred[idx]]),
                'true_class_prob': float(val_probs[idx, val_true[idx]])
            })

        test_misclassified = []
        for idx in test_misclassified_idx:
            test_misclassified.append({
                'sample_index': int(idx),
                'true_label': int(test_true[idx]),
                'predicted_label': int(test_pred[idx]),
                'confidence': float(test_probs[idx, test_pred[idx]]),
                'true_class_prob': float(test_probs[idx, test_true[idx]])
            })

        # Generate classification report
        val_report = classification_report(val_true, val_pred, output_dict=True, zero_division=0)
        test_report = classification_report(test_true, test_pred, output_dict=True, zero_division=0)

        # Generate confusion matrix
        val_cm = confusion_matrix(val_true, val_pred)
        test_cm = confusion_matrix(test_true, test_pred)

        # Compile all results
        analysis = {
            'epoch': metrics['epoch'],
            'train_loss': metrics['train_loss'],
            'validation': {
                'loss': float(metrics['validation']['loss']),
                'accuracy': float(metrics['validation']['accuracy']),
                'f1_micro': float(metrics['validation']['f1_micro']),
                'f1_macro': float(metrics['validation']['f1_macro']),
                'num_samples': len(val_true),
                'num_misclassified': len(val_misclassified_idx),
                'misclassified_samples': val_misclassified,
                'classification_report': val_report,
                'confusion_matrix': val_cm.tolist()
            },
            'test': {
                'loss': float(metrics['test']['loss']),
                'accuracy': float(metrics['test']['accuracy']),
                'f1_micro': float(metrics['test']['f1_micro']),
                'f1_macro': float(metrics['test']['f1_macro']),
                'num_samples': len(test_true),
                'num_misclassified': len(test_misclassified_idx),
                'misclassified_samples': test_misclassified,
                'classification_report': test_report,
                'confusion_matrix': test_cm.tolist()
            }
        }

        # Save to JSON file
        analysis_path = os.path.join(save_path, 'best_model_analysis_1.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\n✓ Best model analysis saved to: {analysis_path}")
        print(f"  Validation misclassified: {len(val_misclassified_idx)}/{len(val_true)} ({100*len(val_misclassified_idx)/len(val_true):.2f}%)")
        print(f"  Test misclassified: {len(test_misclassified_idx)}/{len(test_true)} ({100*len(test_misclassified_idx)/len(test_true):.2f}%)")

        # Also save predictions and probabilities as numpy arrays for further analysis
        np.savez(
            os.path.join(save_path, 'best_model_predictions_1.npz'),
            val_true=val_true,
            val_pred=val_pred,
            val_probs=val_probs,
            test_true=test_true,
            test_pred=test_pred,
            test_probs=test_probs
        )
        print(f"✓ Predictions and probabilities saved to: {os.path.join(save_path, 'best_model_predictions_1.npz')}")
