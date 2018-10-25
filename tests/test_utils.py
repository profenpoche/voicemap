import unittest
import soundfile as sf
import numpy as np
import torch
from torch.nn.modules.distance import CosineSimilarity, PairwiseDistance

from voicemap.utils import *
from config import PATH


class TestWhitening(unittest.TestCase):
    def test_whitening_no_batch(self):
        desired_rms = 0.038021

        test_data, sample_rate = sf.read(PATH + '/data/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac')
        test_data = np.stack([test_data]*2)
        test_data = test_data[:, np.newaxis, :]

        whitened = whiten(torch.from_numpy(test_data), desired_rms)
        # Mean correct
        self.assertTrue(
            np.isclose(whitened.mean().item(), 0),
            'Whitening should reduce mean to 0.'
        )

        # RMS correct
        self.assertTrue(
            np.isclose(np.sqrt(np.power(whitened[0, :], 2).mean()).item(), desired_rms),
            'Whitening should change RMS to desired value.'
        )


class TestDistance(unittest.TestCase):
    def test_query_support_distances(self):
        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two

        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([0, 0])
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 0])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([1, 2])
        support[2] = torch.Tensor([2, 2])

        distances = query_support_distances(query, support, q, k, 'l2')
        self.assertEqual(
            distances.shape, (q * k, k),
            'Output should have shape (q * k, k).'
        )

        # Calculate squared distances by iterating through all query-support pairs
        for i, q_ in enumerate(query):
            for j, s_ in enumerate(support):
                self.assertEqual(
                    (q_ - s_).pow(2).sum(),
                    distances[i, j].item(),
                    'The jth column of the ith row should be the squared distance between the '
                    'ith query sample and the kth query sample'
                )

        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two
        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([1, 0])
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 1])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([-1, -1])
        support[2] = torch.Tensor([0, 2])

        distances = query_support_distances(query, support, q, k, 'cosine')

        # Calculate distances by iterating through all query-support pairs
        for i, q_ in enumerate(query):
            for j, s_ in enumerate(support):
                self.assertTrue(
                    torch.isclose(CosineSimilarity(dim=0)(q_, s_), distances[i, j]),
                    'The jth column of the ith row should be the squared distance between the '
                    'ith query sample and the kth query sample'
                )

    def test_no_nans_on_zero_vectors(self):
        """Cosine distance calculation involves a divide-through by vector magnitude which
        can divide by zeros to occur.
        """
        # Create some dummy data with easily verifiable distances
        q = 1  # 1 query per class
        k = 3  # 3 way classification
        d = 2  # embedding dimension of two
        query = torch.zeros([q * k, d], dtype=torch.double)
        query[0] = torch.Tensor([0, 0])  # First query sample is all zeros
        query[1] = torch.Tensor([0, 1])
        query[2] = torch.Tensor([1, 1])
        support = torch.zeros([k, d], dtype=torch.double)
        support[0] = torch.Tensor([1, 1])
        support[1] = torch.Tensor([-1, -1])
        support[2] = torch.Tensor([0, 0])  # Third support sample is all zeros

        distances = query_support_distances(query, support, q, k, 'cosine')

        self.assertTrue(torch.isnan(distances).sum() == 0, 'Cosine distances between 0-vectors should not be nan')