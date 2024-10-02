import torch
import unittest
from dnaformer.models import RoformerModel

class TestRoformerModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set random seed for reproducibility
        torch.manual_seed(42)

    def setUp(self):
        self.num_tokens = 5  # A, C, G, T, and N
        self.dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.ffn_dim = 256
        self.n_labels = 4  # A, C, G, T
        self.batch_size = 2
        self.seq_length = 100

        self.model = RoformerModel(
            num_tokens=self.num_tokens,
            dim=self.dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim,
            n_labels=self.n_labels
        )

    def test_model_initialization(self):
        self.assertIsInstance(self.model, RoformerModel)
        self.assertEqual(self.model.num_tokens, self.num_tokens)
        self.assertEqual(self.model.dim, self.dim)
        self.assertEqual(self.model.num_heads, self.num_heads)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.ffn_dim, self.ffn_dim)

    def test_forward_pass(self):
        x = torch.randint(0, self.num_tokens, (self.batch_size, self.seq_length))
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.dim))

    def test_output_layer(self):
        x = torch.randint(0, self.num_tokens, (self.batch_size, self.seq_length))
        output = self.model(x)
        logits = self.model.out_logit(output)
        self.assertEqual(logits.shape, (self.batch_size, self.seq_length, self.n_labels))

    def test_long_sequence(self):
        long_seq_length = 1000
        x = torch.randint(0, self.num_tokens, (self.batch_size, long_seq_length))
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, long_seq_length, self.dim))

    def test_single_sequence(self):
        x = torch.randint(0, self.num_tokens, (1, self.seq_length))
        output = self.model(x)
        self.assertEqual(output.shape, (1, self.seq_length, self.dim))

    def test_model_training_mode(self):
        self.model.train()
        x = torch.randint(0, self.num_tokens, (self.batch_size, self.seq_length))
        output = self.model(x)
        self.assertTrue(output.requires_grad)

    def test_model_eval_mode(self):
        self.model.eval()
        x = torch.randint(0, self.num_tokens, (self.batch_size, self.seq_length))
        with torch.no_grad():
            output = self.model(x)
        self.assertFalse(output.requires_grad)

if __name__ == '__main__':
    unittest.main()
