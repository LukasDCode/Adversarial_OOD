import torch
import numpy as np
import torch.nn.functional as F

def calculate_softmax_score(softmax_prediction, num_classes):
    softmax_score_list = []
    softmax_list = list(torch.split(torch.flatten(softmax_prediction), num_classes))
    for softmax_entry in softmax_list:
        softmax_score = F.softmax(softmax_entry.unsqueeze(dim=0), dim=1).data.cpu().numpy()
        softmax_score = np.around(softmax_score[0], 4)
        softmax_score_list.append(softmax_score)
    return softmax_score_list

