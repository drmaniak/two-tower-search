import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, query, pos_doc, neg_doc):
        query_norm = query / query.norm(dim=1, keepdim=True)
        pos_doc_norm = pos_doc / pos_doc.norm(dim=1, keepdim=True)
        neg_doc_norm = neg_doc / neg_doc.norm(dim=1, keepdim=True)

        pos_sim = torch.sum(query_norm * pos_doc_norm, dim=1)
        neg_sim = torch.sum(query_norm * neg_doc_norm, dim=1)

        loss = torch.max(
            neg_sim - pos_sim + self.margin, torch.zeros_like(pos_sim)
        ).mean()
        return loss
