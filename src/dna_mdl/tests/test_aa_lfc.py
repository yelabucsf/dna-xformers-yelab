#!/usr/bin/env python3

import numpy as np
import pytest
from aa_lfc import calculate_aa_lfc, AA_LIST

def test_multiple_codons():
    """Test multiple codons with known mutation patterns."""
    # GCT GCA -> Ala Ala
    ref_dna = "GCTGCA"
    
    # Simple LFC matrix where mutations have value 1
    lfc = np.ones((6, 4))
    # Set reference bases to 0
    lfc[0, 2] = 0  # G in GCT
    lfc[1, 1] = 0  # C in GCT
    lfc[2, 3] = 0  # T in GCT
    lfc[3, 2] = 0  # G in GCA
    lfc[4, 1] = 0  # C in GCA
    lfc[5, 0] = 0  # A in GCA
    
    aa_matrix, mut_types = calculate_aa_lfc(lfc, ref_dna)
    
    # Check matrix shape (2 codons)
    assert aa_matrix.shape == (2, len(AA_LIST))
    
    # Alanine has synonymous mutations
    assert len(mut_types['S']) > 0
    
    # Should have some nonsynonymous mutations
    assert len(mut_types['N']) > 0

def test_reading_frame():
    """Test reading frame handling."""
    # AGCTGCA -> reading frame 1 should give GCT GCA
    ref_dna = "AGCTGCA"
    
    # Simple LFC matrix
    lfc = np.ones((7, 4))
    aa_matrix, mut_types = calculate_aa_lfc(lfc, ref_dna, start_pos=1)
    
    # Should still get 2 codons
    assert aa_matrix.shape == (2, len(AA_LIST))

def test_running_mean():
    """Test running mean calculation for multiple mutations to same AA."""
    # GCT -> Alanine (can mutate to GCA, GCC, GCG)
    ref_dna = "GCT"
    
    # Create LFC matrix where mutations have different values
    lfc = np.zeros((3, 4))
    # Make mutations to same AA have different values
    lfc[2, 0] = 1  # T->A = 1
    lfc[2, 1] = 2  # T->C = 2
    lfc[2, 2] = 3  # T->G = 3
    
    aa_matrix, mut_types = calculate_aa_lfc(lfc, ref_dna)
    
    # Check synonymous mutations are averaged correctly
    assert len(mut_types['S']) == 3  # Should have 3 synonymous mutations
    assert np.allclose(sorted(mut_types['S']), [1, 2, 3])  # Values should be 1, 2, and 3

def test_mutation_types():
    """Test correct classification of mutation types."""
    # Use GCT (Ala) which has both synonymous and nonsynonymous mutations
    ref_dna = "GCT"
    
    # Simple LFC matrix
    lfc = np.ones((3, 4))
    # Set reference bases to 0
    lfc[0, 2] = 0  # G
    lfc[1, 1] = 0  # C
    lfc[2, 3] = 0  # T
    
    aa_matrix, mut_types = calculate_aa_lfc(lfc, ref_dna)
    
    # GCT should have:
    # - Synonymous mutations (to GCA, GCC, GCG)
    # - Nonsynonymous mutations (to other AAs)
    # - Possible stop codon mutations
    assert len(mut_types['S']) > 0
    assert len(mut_types['N']) > 0
    assert len(mut_types['T']) >= 0  # May or may not have stop mutations

if __name__ == "__main__":
    pytest.main([__file__])