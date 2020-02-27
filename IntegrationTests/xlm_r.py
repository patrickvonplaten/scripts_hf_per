import torch
import ipdb

xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()
ipdb.set_trace()
