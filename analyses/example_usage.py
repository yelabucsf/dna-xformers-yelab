import torch
from dnaformer.models import RoformerModel
from dnaformer.tokenizers import DNAKmerTokenizer

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize tokenizer
    tokenizer = DNAKmerTokenizer(k=3)  # Using 3-mers for this example

    # Example DNA sequences
    dna_sequences = [
        "ATGCATGCATGC",
        "GCTAGCTAGCTA",
        "TTTTAAAACCCC",
    ]

    # Tokenize sequences
    tokenized_seqs = tokenizer.tokenize_2_ixs(dna_sequences)
    input_ids = tokenized_seqs[0]  # The first element contains the token IDs

    # Model parameters
    num_tokens = tokenizer.num_tokens
    dim = 128
    num_heads = 4
    num_layers = 3
    ffn_dim = 256
    n_labels = 4  # For predicting A, C, G, T

    # Initialize model
    model = RoformerModel(
        num_tokens=num_tokens,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        n_labels=n_labels
    )

    # Set model to evaluation mode
    model.eval()

    # Perform forward pass
    with torch.no_grad():
        outputs = model(input_ids)

    # Process outputs
    logits = model.out_logit(outputs)
    predictions = torch.argmax(logits, dim=-1)

    # Print results
    print("Input Sequences:")
    for seq in dna_sequences:
        print(seq)

    print("\nModel Outputs:")
    print(f"Output shape: {outputs.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")

    print("\nPredicted Nucleotides:")
    nucleotides = ['A', 'C', 'G', 'T']
    for i, pred in enumerate(predictions):
        pred_seq = ''.join([nucleotides[p.item()] for p in pred])
        print(f"Sequence {i + 1}: {pred_seq}")

if __name__ == "__main__":
    main()