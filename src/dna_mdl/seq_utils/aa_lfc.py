#!/usr/bin/env python3

from typing import Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray

# Codon to amino acid mapping
CODON2AA = {
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "AGA": "R", "AGG": "R", "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GAC": "D", "GAT": "D",
    "AAC": "N", "AAT": "N",
    "TGC": "C", "TGT": "C",
    "GAA": "E", "GAG": "E",
    "CAA": "Q", "CAG": "Q",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "CAC": "H", "CAT": "H",
    "ATA": "I", "ATC": "I", "ATT": "I",
    "TTA": "L", "TTG": "L", "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "AAA": "K", "AAG": "K",
    "ATG": "M",
    "TTC": "F", "TTT": "F",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "AGC": "S", "AGT": "S", "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "TGG": "W",
    "TAC": "Y", "TAT": "Y",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "TAA": "STOP", "TAG": "STOP", "TGA": "STOP"
}

NUCLEOS = ["A", "C", "G", "T"]
AA_LIST = sorted(list(set(CODON2AA.values())))
AA_LIST.remove("STOP")  # Remove STOP from amino acid matrix columns

def validate_dna_sequence(seq: str) -> bool:
    """Validate DNA sequence contains only valid nucleotides."""
    return all(n in NUCLEOS for n in seq.upper())

def validate_inputs(lfc_matrix: NDArray, ref_dna: str, start_pos: int) -> None:
    """Validate input parameters."""
    if not isinstance(lfc_matrix, np.ndarray):
        raise TypeError("lfc_matrix must be a numpy array")
    
    if lfc_matrix.ndim != 2 or lfc_matrix.shape[1] != 4:
        raise ValueError("lfc_matrix must have shape (N, 4)")
        
    if start_pos not in (0, 1, 2):
        raise ValueError("start_pos must be 0, 1, or 2")
        
    if not validate_dna_sequence(ref_dna):
        raise ValueError("ref_dna contains invalid nucleotides")
        
    if len(ref_dna) != lfc_matrix.shape[0]:
        raise ValueError("ref_dna length must match lfc_matrix rows")

def get_nucleotide_index(nuc: str) -> int:
    """Convert nucleotide to index."""
    return NUCLEOS.index(nuc.upper())

def calculate_aa_lfc(
    lfc_matrix: NDArray, 
    ref_dna: str,
    start_pos: int = 0
) -> Tuple[NDArray, Dict[str, List[float]]]:
    """
    Calculate amino acid log fold changes from nucleotide substitution matrix.
    
    Args:
        lfc_matrix: Nx4 matrix of log fold changes for each position, with columns
                   corresponding to A,C,G,T in that order. Each value represents
                   the log fold change for mutating to that nucleotide.
        ref_dna: Reference DNA sequence string containing only A,C,G,T
        start_pos: Starting position (0,1,2) for reading frame
    
    Returns:
        aa_matrix: Mx20 matrix of average LFCs per amino acid where M is the number
                  of complete codons. Each row corresponds to a codon position, and
                  columns correspond to amino acids in alphabetical order (excluding STOP).
                  Values are means of all possible mutations leading to that amino acid.
        mut_types: Dictionary with mutation type statistics containing:
                  'S': synonymous mutations (same amino acid)
                  'N': nonsynonymous mutations (different amino acid)
                  'T': terminating mutations (to stop codon)
    
    Notes:
        - Input matrix should have shape (N, 4) with columns corresponding to A,C,G,T
        - Output aa_matrix will have shape (M, 20) where M is number of complete codons
        - Positions that cannot form complete codons are trimmed
        - NA values indicate impossible amino acid transitions
        - STOP codons are not included in aa_matrix but are tracked in mut_types['T']
        - Mean values for amino acid changes are calculated as running means to
          avoid bias from mutation order
    
    Example:
        For a DNA sequence "ATGGCT" and start_pos=0:
        - Will process 2 complete codons: ATG (M) and GCT (A)
        - aa_matrix will have shape (2, 20)
        - First row contains average LFCs for mutations from ATG
        - Second row contains average LFCs for mutations from GCT
    """
    # Validate inputs
    validate_inputs(lfc_matrix, ref_dna, start_pos)
    ref_dna = ref_dna.upper()
    
    # Trim incomplete codons at start and end
    seq_len = len(ref_dna) - start_pos
    n_codons = seq_len // 3
    if n_codons == 0:
        raise ValueError("Sequence too short to form complete codons")
    
    # Trim sequence and LFC matrix to complete codons
    end_pos = start_pos + (3 * n_codons)
    ref_dna = ref_dna[start_pos:end_pos]
    lfc_matrix = lfc_matrix[start_pos:end_pos]
    
    # Reshape into codon structure
    lfc_codons = lfc_matrix.reshape(-1, 3, 4)
    ref_codons = [ref_dna[i:i+3] for i in range(0, len(ref_dna), 3)]
    
    # Initialize tracking arrays
    aa_sums = np.zeros((n_codons, len(AA_LIST)))
    aa_counts = np.zeros((n_codons, len(AA_LIST)))
    mut_types = {"S": [], "N": [], "T": []}
    
    # Process each codon
    for i, ref_codon in enumerate(ref_codons):
        ref_aa = CODON2AA[ref_codon]
        codon_lfc = lfc_codons[i]
        
        # Calculate all possible single mutations
        for pos in range(3):
            ref_nuc = ref_codon[pos]
            ref_nuc_idx = get_nucleotide_index(ref_nuc)
            
            for mut_nuc_idx, mut_nuc in enumerate(NUCLEOS):
                if mut_nuc == ref_nuc:
                    continue
                    
                # Create mutated codon
                mut_codon = (
                    ref_codon[:pos] + 
                    mut_nuc + 
                    ref_codon[pos+1:]
                )
                mut_aa = CODON2AA[mut_codon]
                
                # Get LFC for this mutation
                lfc = codon_lfc[pos, mut_nuc_idx]
                
                # Categorize mutation
                if mut_aa == ref_aa:
                    mut_types["S"].append(lfc)
                elif mut_aa == "STOP":
                    mut_types["T"].append(lfc)
                else:
                    mut_types["N"].append(lfc)
                    
                # Add to amino acid sums and counts
                if mut_aa != "STOP":
                    aa_idx = AA_LIST.index(mut_aa)
                    aa_sums[i, aa_idx] += lfc
                    aa_counts[i, aa_idx] += 1
    
    # Calculate means, handling division by zero
    aa_matrix = np.divide(
        aa_sums, 
        aa_counts, 
        out=np.full_like(aa_sums, np.nan),
        where=aa_counts > 0
    )
    
    return aa_matrix, mut_types
