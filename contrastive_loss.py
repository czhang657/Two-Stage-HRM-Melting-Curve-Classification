import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss

    Based on: https://arxiv.org/abs/2004.11362

    Args:
        temperature: Temperature parameter for scaling
        base_temperature: Base temperature (usually same as temperature)
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: shape [batch_size, feature_dim]
            labels: shape [batch_size]

        Returns:
            loss: scalar tensor
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        # shape: [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log_prob
        # Divide by temperature
        logits = similarity_matrix / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss