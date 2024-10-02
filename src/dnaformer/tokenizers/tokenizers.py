#!/usr/bin/env python3
#
# Tokenizers

from typing import List, Union, Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from itertools import product


def generate_combinations(characters, length):
    # Convert the set of characters to a sorted list
    char_list = sorted(list(characters))

    # Generate all possible combinations
    combinations = ["".join(combo) for combo in product(char_list, repeat=length)]

    # Create a dictionary with combinations as keys and indices as values
    result = {combo: index for index, combo in enumerate(combinations)}

    return result


class DNA1merTokenizer:
    """
    A simple character-level tokenizer for DNA sequences.

    This tokenizer converts DNA sequences into numerical tokens, where each nucleotide
    is assigned a unique integer. It also handles special tokens such as padding,
    masking, separation, and unknown tokens.

    Parameters:
    -----------
    model_n : bool, optional (default=False)
        If True, includes 'N' in the nucleotide tokens (A, C, G, T, N).
        If False, only uses A, C, G, T.

    Attributes:
    -----------
    n_tokens : int
        The total number of tokens, including special tokens and nucleotides.
    token_ix2str : dict
        Mapping from token indices to their string representations.
    token_str2ix : dict
        Mapping from token strings to their indices.
    spec_tokens : list of str
        List of special tokens: ["<PAD>", "<MASK>", "<SEP>", "<CLS>", "<UNK>"]
    pad_token : str
        The padding token ("<PAD>").
    mask_token : str
        The masking token ("<MASK>").
    sep_token : str
        The separator token ("<SEP>").
    cls_token : str
        The classifier token ("<CLS>").
    unk_token : str
        The unknown token ("<UNK>").
    model_n : bool
        Whether 'N' is included in the nucleotide tokens.
    nucleos : list of str
        List of nucleotide tokens (["A", "C", "G", "T"] or ["A", "C", "G", "T", "N"]).

    Methods:
    --------
    encode(seq: str, return_tensor: bool = False) -> Union[List[int], Tensor]:
        Encode a DNA sequence into a list of token indices.

    decode(tokens: Union[List[int], Tensor]) -> str:
        Decode a list of token indices back into a DNA sequence.

    add_special_tokens(tokens: Tensor, add_cls: bool = True, add_sep: bool = True) -> Tensor:
        Add special tokens (CLS and/or SEP) to the input tensor.

    tokix_to_label(tokens: Tensor) -> Tensor:
        Convert token indices to nucleotide labels.

    tokenize_seqs(input: List[str]) -> torch.Tensor:
        Tokenize a list of input sequences, padding to the maximum sequence length.

    Examples:
    ---------
    >>> tokenizer = DNA1merTokenizer(model_n=True)
    >>> seq = "ACGTNA"
    >>> tokens = tokenizer.encode(seq)
    >>> print(tokens)
    [5, 6, 7, 8, 9, 5]
    >>> decoded_seq = tokenizer.decode(tokens)
    >>> print(decoded_seq)
    ACGTNA
    """
    def __init__(self, model_n: bool = False):
        """
        Parameters
        ----------
        model_n : bool
            Include `N` in the nucleotide tokens (A, C, G, T, N).
        """
        self.n_tokens: int = 0

        self.token_ix2str: dict[int, str] = {}
        self.token_str2ix: dict[str, int] = {}

        # add special tokens
        self.spec_tokens: List[str] = ["<PAD>", "<MASK>", "<SEP>", "<CLS>", "<UNK>"]
        self.pad_token: str = "<PAD>"
        self.mask_token: str = "<MASK>"
        self.sep_token: str = "<SEP>"
        self.cls_token: str = "<CLS>"
        self.unk_token: str = "<UNK>"
        for i, token in enumerate(self.spec_tokens):
            self.token_ix2str[i] = token
            self.token_str2ix[token] = i
            self.n_tokens += 1

        # nucleotide tokens
        self.model_n: bool = model_n
        self.nucleos: List[str] = ["A", "C", "G", "T"]
        if model_n:
            self.nucleos.append("N")

        # add 1mer tokens (A, C, G, T, N)
        for i in range(len(self.nucleos)):
            self.token_ix2str[self.n_tokens] = self.nucleos[i]
            self.token_str2ix[self.nucleos[i]] = self.n_tokens
            self.n_tokens += 1

    def encode(self, seq: str, return_tensor: bool = False) -> Union[List[int], Tensor]:
        """
        Encode a DNA sequence into a list of token indices.

        Parameters:
        -----------
        seq : str
            The input DNA sequence to encode.
        return_tensor : bool, optional (default=False)
            If True, return a PyTorch tensor; if False, return a list.

        Returns:
        --------
        Union[List[int], Tensor]
            The encoded sequence as a list of integers or a PyTorch tensor.

        Notes:
        ------
        If model_n is False and 'N' is encountered in the sequence, it will be
        randomly replaced with A, C, G, or T.
        """
        seq = seq.upper()
        if not self.model_n:
            rng = np.random.default_rng()
            b_l = ["A", "C", "G", "T"]
            seq_l = [rng.choice(b_l) if base == "N" else base for base in seq]
            seq = "".join(seq_l)

        tokens: Union[List[int], Tensor] = [self.token_str2ix[base] for base in seq]
        if return_tensor:
            tokens = torch.tensor(tokens)
        return tokens

    def decode(self, tokens: Union[List[int], Tensor]) -> str:
        """
        Decode a list of token indices back into a DNA sequence.

        Parameters:
        -----------
        tokens : Union[List[int], Tensor]
            The list of token indices or a PyTorch tensor to decode.

        Returns:
        --------
        str
            The decoded DNA sequence.

        Raises:
        -------
        ValueError
            If the input tensor has more than one dimension.
        """
        if isinstance(tokens, Tensor):
            if tokens.ndim > 1:
                raise ValueError("tokens should be 1D")
            tokens = tokens.tolist()
        seq: str = "".join([self.token_ix2str[ix] for ix in tokens])
        return seq

    def add_special_tokens(
        self, tokens: Tensor, add_cls: bool = True, add_sep: bool = True
    ) -> Tensor:
        """
        Add special tokens (CLS and/or SEP) to the input tensor.

        Parameters:
        -----------
        tokens : Tensor
            The input tensor of token indices.
        add_cls : bool, optional (default=True)
            If True, prepend the CLS token.
        add_sep : bool, optional (default=True)
            If True, append the SEP token.

        Returns:
        --------
        Tensor
            The input tensor with added special tokens.
        """
        cls_token: int = self.token_str2ix[self.cls_token]
        sep_token: int = self.token_str2ix[self.sep_token]
        if tokens.ndim > 1:
            n_seq: int = tokens.shape[0]
        else:
            n_seq = 1
        if add_cls:
            cls_tensor: Tensor = torch.full((n_seq, 1), cls_token)
            tokens = torch.cat([cls_tensor, tokens], dim=-1)
        if add_sep:
            sep_tensor: Tensor = torch.full((n_seq, 1), sep_token)
            tokens = torch.cat([tokens, sep_tensor], dim=-1)
        return tokens

    def tokix_to_label(self, tokens: Tensor) -> Tensor:
        """
        Convert token indices to nucleotide labels.

        Parameters:
        -----------
        tokens : Tensor
            The input tensor of token indices.

        Returns:
        --------
        Tensor
            A tensor of nucleotide labels (0 for A, 1 for C, 2 for G, 3 for T, 4 for N if model_n is True).
        """
        labels = tokens.clone()
        for i, bp in enumerate(self.nucleos):
            labels[labels == self.token_str2ix[bp]] = i
        return labels

    def tokenize_seqs(self, input: List[str]) -> torch.Tensor:
        """
        Tokenize a list of input sequences, padding to the maximum sequence length.

        Parameters:
        -----------
        input : List[str]
            A list of input DNA sequences to tokenize.

        Returns:
        --------
        torch.Tensor
            A tensor of tokenized and padded sequences, shape (batch_size, max_seq_len).
        """
        inputs = [torch.tensor(self.encode(seq)) for seq in input]
        pad_ix = self.token_str2ix[self.pad_token]
        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=pad_ix
        )
        return inputs


class DNAKmerTokenizer:
    """
    Noncontiguous Kmer tokenizer for DNA sequences with single nucleotide labels.

    This tokenizer converts DNA sequences into k-mer tokens, supporting masking and special token handling.
    It's designed for use in masked language modeling tasks for DNA sequences.

    Args:
        k (int, optional): The k-mer size. Defaults to 5.
        add_cls (bool, optional): Whether to add a CLS token at the beginning. Defaults to True.
        add_sep (bool, optional): Whether to add a SEP token at the end. Defaults to True.
        mask_prob (float, optional): Probability of masking a token. Defaults to 0.15.
        prob_keep (float, optional): Probability of keeping a masked token unchanged. Defaults to 0.1.
        prob_rand (float, optional): Probability of replacing a masked token with a random token. Defaults to 0.0.

    Attributes:
        k (int): The k-mer size.
        add_cls (bool): Whether to add a CLS token.
        add_sep (bool): Whether to add a SEP token.
        mask_prob (float): Masking probability.
        prob_keep (float): Probability of keeping a masked token unchanged.
        prob_rand (float): Probability of replacing a masked token with a random token.
        token_ix2str (dict): Mapping from token indices to strings.
        token_str2ix (dict): Mapping from token strings to indices.
        cls_token (str): The CLS token string.
        sep_token (str): The SEP token string.
        mask_token (str): The MASK token string.
        pad_token (str): The PAD token string.
        nucleos (list): List of nucleotide characters.
        kmer_str2ix (dict): Mapping from k-mer strings to indices.

    Methods:
        str2arr(seq: str) -> np.ndarray:
            Convert a DNA sequence to a numpy array of nucleotides.
         
        strs2arr(seqs: List[str], pad: bool = True) -> np.ndarray:
            Convert a list of DNA sequences to a numpy string array of nucleotides.
         
        arr_mask(arr: np.ndarray, ...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Mask the numpy string array for masked language modeling.
         
        tokenize_to_strs(input: List[str], ...) -> List[np.ndarray]:
            Tokenize input sequences to string arrays with masking.
         
        tokenize_2_ixs(input: List[str], ...) -> Union[List[Tensor], List[np.ndarray]]:
            Tokenize input sequences to token indices with masking.

    Note:
        - This tokenizer is specifically designed for DNA sequences and k-mer based tokenization.
        - It supports masking operations for masked language modeling tasks.
        - The tokenizer can handle special tokens (CLS, SEP) and padding.
        - It provides both string-based and index-based tokenization methods.
    """
    def __init__(self, k: int = 5, add_cls: bool = True, add_sep: bool = True,
                 mask_prob: float = 0.15, prob_keep: float = 0.1, prob_rand: float = 0.0):
        """
        Parameters
        ----------
        """
        self.n_tokens: int = 0
        self.k: int = k
        self.add_cls: bool = add_cls
        self.add_sep: bool = add_sep
        self.mask_prob: float = mask_prob
        self.prob_keep: float = prob_keep
        self.prob_rand: float = prob_rand

        self.token_ix2str: dict[int, str] = {}
        self.token_str2ix: dict[str, int] = {}

        # add special tokens
        self.cls_token: str = "!"
        self.cls_ix: int = 0
        self.sep_token: str = "@"
        self.sep_ix: int = 1
        self.mask_token: str = "N"
        self.pad_token: str = "N"
        self.spec_tokens: List[str] = ["!", "@", "N"]
        for i, token in enumerate(self.spec_tokens):
            self.token_ix2str[i] = token
            self.token_str2ix[token] = i
            self.n_tokens += 1

        # nucleotide tokens
        self.nucleos: List[str] = ["A", "C", "G", "T"]

        # kmer elements
        self.ke = self.nucleos.copy() + list(set([self.pad_token, self.mask_token]))
        self.ke_ix2str: dict[int, str] = {}
        self.ke_str2ix: dict[str, int] = {}
        for i in range(len(self.ke)):
            self.ke_ix2str[i] = self.ke[i]
            self.ke_str2ix[self.ke[i]] = i

        self.ke_str2ixv: np.vectorize = np.vectorize(self.ke_str2ix.get)
        self.num_tokens = 2 + len(self.ke) ** k

        self.kmer_str2ix: dict[str, int] = generate_combinations(self.ke, k)
        self.kmer_base: np.ndarray = len(self.ke) ** np.arange(k)[::-1]

    def kmer2ix(self, kmer: np.ndarray) -> int:
        if kmer.ndim != 1:
            raise ValueError(f"kmer should be 1D, not {kmer.ndim}D: {kmer}")
        if kmer.shape[0] != self.k:
            raise ValueError(f"kmer should be length {self.k}")
        ix = self.kmer_str2ix["".join(kmer)]
        return ix + 2

    def kmer2ix_np(self, kmer: np.ndarray) -> np.ndarray:
        if kmer.shape[-1] != self.k:
            raise ValueError(f"kmer should be length {self.k} along last dimension")
        kmer_ix = np.sum(self.ke_str2ixv(kmer) * self.kmer_base, axis=-1,
                         dtype=np.int32, keepdims=True)
        return kmer_ix + 2

    def ix2kmer(self, ix: int) -> np.ndarray:
        kmer: np.ndarray = np.full(self.k, "N", dtype="U6")
        nke = len(self.ke)
        ix -= 2
        for i in range(self.k):
            k_ix = ix % nke
            kmer[i] = self.ke_ix2str[k_ix]
            ix = ix // nke
        return kmer

    def str2arr(self, seq: str) -> np.ndarray:
        """
        Convert a DNA sequence to a numpy array of nucleotides
        """
        seq = seq.upper()
        arr = np.frombuffer(seq.encode("ascii"), dtype="S1").astype("U6")
        return arr

    def strs2arr(self, seqs: List[str], pad: bool = True) -> np.ndarray:
        """
        Convert a list of DNA sequences to a numpy string array of nucleotides.
        """
        arr_l = [self.str2arr(seq) for seq in seqs]
        max_len = max([len(arr) for arr in arr_l])
        if pad:
            pad_tok = self.pad_token
            for i, arr in enumerate(arr_l):
                if len(arr) < max_len:
                    pad_len = max_len - len(arr)
                    pad_arr = np.full(pad_len, pad_tok)
                    arr_l[i] = np.concatenate([arr, pad_arr])
        arr = np.array(arr_l)
        return arr

    def arr_add_special_tok(self, arr: np.ndarray) -> np.ndarray:
        """
        Add special tokens to the string array.
        """
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        B: int = arr.shape[0]
        if self.add_cls:
            cls_tok: str = self.cls_token
            cls_arr: np.ndarray = np.full((B, 1), cls_tok)
            arr = np.concatenate([cls_arr, arr], axis=1)
        if self.add_sep:
            sep_tok: str = self.sep_token
            sep_arr: np.ndarray = np.full((B, 1), sep_tok)
            arr = np.concatenate([arr, sep_arr], axis=1)

        return arr

    def arr_mask(
        self,
        arr: np.ndarray,
        mask_prob: Optional[float] = None,
        prob_keep: Optional[float] = None,
        prob_rand: Optional[float] = None,
        prob_scale: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mask the numpy string array.
        Return masked char array, boolean of masked indices, and integer label array.
        """

        if mask_prob is None:
            mask_prob = self.mask_prob
        if prob_keep is None:
            prob_keep = self.prob_keep
        if prob_rand is None:
            prob_rand = self.prob_rand

        mask_probs = np.full(arr.shape, mask_prob)

        # remove special tokens from mask_probs
        for sp in self.spec_tokens:
            mask_probs[arr == sp] = 0.0

        # scale probabilities by given array.
        if prob_scale is not None:
            mask_probs *= prob_scale

        keep_probs = np.full(arr.shape, prob_keep)
        rand_probs = np.full(arr.shape, prob_rand)

        rng = np.random.default_rng()
        mask_ixs = rng.binomial(1, mask_probs).astype(bool)

        keep_ixs = rng.binomial(1, keep_probs).astype(bool)
        keep_ixs = keep_ixs & mask_ixs

        rand_ixs = rng.binomial(1, rand_probs).astype(bool)
        rand_ixs = rand_ixs & (~keep_ixs)
        rand_ixs = rand_ixs & mask_ixs

        replace_ix = mask_ixs & ~keep_ixs

        # replace values in array copy
        mskd: np.ndarray = arr.copy()

        mskd[replace_ix] = self.mask_token

        n_rand: int = rand_ixs.sum()
        mskd[rand_ixs] = rng.choice(self.nucleos, size=n_rand)

        # get labels
        labels: np.ndarray = np.full(arr.shape, -100)
        for base in self.nucleos:
            labels[arr == base] = self.ke_str2ix[base]

        return mskd, mask_ixs, labels

    def pad_arr(self, arr, pad_val=None, add_cls=None, add_sep=None) -> np.ndarray:
        """
        Pad the string array.
        """
        start: int = 0
        end: int = arr.shape[-1]
        if add_cls is None:
            add_cls = self.add_cls
        if add_sep is None:
            add_sep = self.add_sep
        if add_cls:
            start += 1
        if add_sep:
            end -= 1
        T: int = end - start

        if pad_val is None:
            pad_val = self.pad_token

        pad_len: int = (self.k - T % self.k) % self.k
        arr_p: np.ndarray = np.full(arr.shape[:-1] + (pad_len,), pad_val)
        arr_c: np.ndarray = np.concatenate([arr[:, :end], arr_p], axis=-1)
        if add_sep:
            arr_c = np.append(arr_c, arr[..., -1:], axis=-1)

        return arr_c

    def pad_arrs_k(
        self, mskd: np.ndarray, mask_ixs: np.ndarray, labels: np.ndarray
    ) -> List[np.ndarray]:
        mskd = self.pad_arr(mskd, self.pad_token)
        mask_ixs = self.pad_arr(mask_ixs, False)
        labels = self.pad_arr(labels, -100)

        return [mskd, mask_ixs, labels]

    def arr2ix(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert token strings to kmer indices
        Input `arr` is of shape `(B, L)`
        """
        start: int = 0
        end: int = arr.shape[-1]
        if self.add_cls:
            start += 1
        if self.add_sep:
            end -= 1
        T: int = end - start
        B: int = arr.shape[0]

        if T % self.k != 0:
            raise ValueError(f"Sequence length {T} not divisible by k={self.k}")

        # kmer_arr is (B, n_kmer, k)
        n_kmer = T // self.k
        kmer_arr = arr[:, start:end].reshape(B, n_kmer, self.k)

        # output is (B, n_kmer)
        output = self.kmer2ix_np(kmer_arr)
        output = output.squeeze(2)

        if self.add_cls:
            cls_arr: np.ndarray = np.full((B, 1), self.cls_ix)
            output = np.append(cls_arr, output, axis=1)
        if self.add_sep:
            sep_arr: np.ndarray = np.full((B, 1), self.sep_ix)
            output = np.append(output, sep_arr, axis=1)

        return output

    def tokenize_to_strs(
        self,
        input: List[str],
        mask_prob: float = 0.15,
        prob_keep: float = 0.1,
        prob_rand: float = 0.0,
        prob_scale: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Tokenize input sequences. Pads up to max sequence length.

        Parameters
        ----------
        input : List[str]
            List of input sequences.

        Returns
        -------
        torch.Tensor
            Tokenized input sequences of shape (batch, seq_len).
        """
        arr: np.ndarray = self.strs2arr(input)
        arr = self.arr_add_special_tok(arr)
        mskd, mask_ixs, labels = self.arr_mask(
            arr,
            mask_prob=mask_prob,
            prob_keep=prob_keep,
            prob_rand=prob_rand,
            prob_scale=prob_scale,
        )
        mskd, mask_ixs, labels = self.pad_arrs_k(mskd, mask_ixs, labels)
        return [mskd, mask_ixs, labels]

    def tokenize_2_ixs(
        self,
        input: List[str],
        mask_prob: float = 0.15,
        prob_keep: float = 0.1,
        prob_rand: float = 0.0,
        prob_scale: Optional[np.ndarray] = None,
        return_tensor: bool = True,
    ) -> Union[List[Tensor], List[np.ndarray]]:
        """
        Tokenize input sequences. Pads up to max sequence length.

        Parameters
        ----------
        input : List[str]
            List of input sequences.

        Returns
        -------
        torch.Tensor
            Tokenized input sequences of shape (batch, seq_len).
        """
        mskd, mask_ixs, labels = self.tokenize_to_strs(
            input, mask_prob, prob_keep, prob_rand, prob_scale
        )
        kmer_ix: np.ndarray = self.arr2ix(mskd)
        if return_tensor:
            kmer_ix_t = torch.tensor(kmer_ix, dtype=torch.long)
            mask_ixs_t = torch.tensor(mask_ixs, dtype=torch.bool)
            labels_t = torch.tensor(labels, dtype=torch.long)
            return [kmer_ix_t, mask_ixs_t, labels_t]
        else:
            return [kmer_ix, mask_ixs, labels]
