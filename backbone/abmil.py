import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, attention_dim=128, num_classes=8, dropout=0.25):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input, returnt='out', inst_feat=False):
        x = input[0]
        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        if x.dim() != 2:
            raise ValueError('ABMIL expects flat bag features with shape [N, D], got {}'.format(tuple(x.shape)))

        h = self.feature_proj(x)
        attn_logits = self.attention(h).transpose(0, 1)
        attn = F.softmax(attn_logits, dim=1)
        bag_feature = torch.mm(attn, h)

        if returnt == 'features':
            return bag_feature

        logits = self.classifier(bag_feature)
        y_prob = F.softmax(logits, dim=1)
        y_hat = torch.argmax(logits, dim=1)
        sim_loss = logits.sum() * 0.0

        return logits, y_prob, y_hat, attn.squeeze(0), sim_loss
