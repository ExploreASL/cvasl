import torch
import torch.nn as nn

class InfoNCE(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_view1, embeddings_view2):
        """
        Calculates the InfoNCE loss.

        Args:
            embeddings_view1 (torch.Tensor): Embeddings for first view of each image (B, D).
            embeddings_view2 (torch.Tensor): Embeddings for second view of each image (B, D).

        Returns:
            torch.Tensor: InfoNCE loss.
        """
        batch_size = embeddings_view1.shape[0]
        embeddings = torch.cat([embeddings_view1, embeddings_view2], dim=0) # (2B, D)
        similarity_matrix = nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2) # (2B, 2B)

        # Mask out self-similarity (diagonal) - not needed as we are comparing view1 to view2
        # mask = torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)
        # similarity_matrix.masked_fill_(mask, -1e9)

        # Positive pairs are view1[i] and view2[i] for each i in batch
        positives = torch.diag(similarity_matrix, batch_size) # view1[i] vs view2[i]  (B,)
        negatives_mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device) # exclude same index pairs within view1/view2
        negatives = similarity_matrix[negatives_mask].reshape(2*batch_size, -1) # (2B, 2B-2)

        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) # (B, 2B-1)
        logits /= self.temperature

        labels = torch.zeros(batch_size, dtype=torch.long, device=embeddings.device) # positive pair is always the first element
        loss = nn.functional.cross_entropy(logits, labels)
        return loss