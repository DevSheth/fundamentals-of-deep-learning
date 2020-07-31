import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal_

class glove(nn.Module):
    def __init__(self, V, K):
        super(glove, self).__init__()
        self.V = V
        self.K = K
        self.W = nn.Parameter(torch.randn(V, K))
        self.B = nn.Parameter(torch.randn(V))
        
    def forward(self, i, j):
        return torch.dot(self.W[i], self.W[j]) + self.B[i] + self.B[j]
    
def f(x, x_max, alpha):
    if(x < x_max):
        return (x/x_max)**alpha
    else:
        return 1

class GloVe(nn.Module):
    def __init__(self, co_oc, embed_size, x_max=100, alpha=0.75):
        """
        :param co_oc: Co-occurrence ndarray with shape of [num_classes, num_classes]
        :param embed_size: embedding size
        :param x_max: An int representing cutoff of the weighting function
        :param alpha: Ant float parameter of the weighting function
        """

        super(GloVe, self).__init__()

        self.embed_size = embed_size
        self.x_max = x_max
        self.alpha = alpha

        ''' co_oc Matrix is shifted in order to prevent having log(0) '''
        self.co_oc = co_oc + 1.0

        [self.num_classes, _] = self.co_oc.shape

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.in_embed.weight = xavier_normal_(self.in_embed.weight)

        self.in_bias = nn.Embedding(self.num_classes, 1)
        self.in_bias.weight = xavier_normal_(self.in_bias.weight)

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size)
        self.out_embed.weight = xavier_normal_(self.out_embed.weight)

        self.out_bias = nn.Embedding(self.num_classes, 1)
        self.out_bias.weight = xavier_normal_(self.out_bias.weight)

    def forward(self, input, output):
        """
        :param input: An array with shape of [batch_size] of int type
        :param output: An array with shape of [batch_size] of int type
        :return: loss estimation for Global Vectors word representations
                 defined in nlp.stanford.edu/pubs/glove.pdf
        """

        batch_size = len(input)

        co_occurences = np.array([self.co_oc[input[i], output[i]] for i in range(batch_size)])
        weights = np.array([self._weight(var) for var in co_occurences])

        co_occurences = Variable(torch.from_numpy(co_occurences)).float()
        weights = Variable(torch.from_numpy(weights)).float()

        input = Variable(torch.from_numpy(input)).type(torch.LongTensor)
        output = Variable(torch.from_numpy(output)).type(torch.LongTensor)

        input_embed = self.in_embed(input)
        input_bias = self.in_bias(input)
        output_embed = self.out_embed(output)
        output_bias = self.out_bias(output)

        return (torch.pow(
            ((input_embed * output_embed).sum(1) + input_bias + output_bias).squeeze(1) - torch.log(co_occurences), 2
        ) * weights).sum()

    def _weight(self, x):
        return 1 if x > self.x_max else (x / self.x_max) ** self.alpha

    def embeddings(self):
        return self.in_embed.weight.data.cpu().numpy() + self.out_embed.weight.data.cpu().numpy()
