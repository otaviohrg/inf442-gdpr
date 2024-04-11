from gdpr import MLP
from gdpr import logits_to_probs, probs_to_max_classes, logits_to_cross_entropy_loss
import torch

x = torch.rand([10, 5])
targs = torch.randint(0, 9, (1, 10)).squeeze(dim=0)
print(x)
print(targs)
model = MLP(5, 5, 1, 10)
y = model(x)
print(y)
print(logits_to_probs(y))
print(probs_to_max_classes(logits_to_probs(y)))
print(logits_to_cross_entropy_loss(y, targs))