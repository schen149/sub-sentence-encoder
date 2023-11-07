"""Helper classes/functions for proposition pair labels."""
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import numpy as np
from scipy.signal import convolve2d
from scipy.optimize import linear_sum_assignment

import torch
from typing import List, Tuple

tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


class PropPairLabels:
    """
    Label for proposition pairs
    """
    POSITIVE: str = "p"
    NEGATIVE_CONTRADICT: str = "n_con"
    NEGATIVE_INTRA_SENTENCE: str = "n_in"


quote_translation = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])


def normalize_text(text: str):
    text = text.translate(quote_translation)
    return text


def tokenize_text(text):
    token_spans = list(tokenizer.span_tokenize(text))
    tokens = [text[s:e] for s, e in token_spans]
    return tokens, token_spans


def lemmatize_tokens(tokens):
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def indices_to_span(indices):
    indices = sorted(indices)
    spans = []
    cur_span_ids = []
    for idx in indices:
        if not cur_span_ids:
            cur_span_ids.append(idx)
        elif cur_span_ids[-1] == idx - 1:
            cur_span_ids.append(idx)
        else:
            spans.append((cur_span_ids[0], cur_span_ids[-1] + 1))
            cur_span_ids = [idx]

    if len(cur_span_ids) > 0:
        spans.append((cur_span_ids[0], cur_span_ids[-1] + 1))

    return spans


def check_alignment_length(decomp_sent: str,
                           aligned_spans: List[List[int]],
                           len_th: float = 0.5):
    """
    This function is used to check if the total length of aligned spans in a given decomposition sentence meets a certain threshold based on the ratio of sentence-to-alignment length.
    :param decomp_sent: The decomposition sentence to be checked for alignment length.
    :param aligned_spans: A list of aligned spans represented as pairs of indices [start, end] indicating the positions of aligned words in the decomposition sentence.
    :param len_th: The ratio of sentence-to-alignment length that a valid alignment needs to meet. Default value is 0.5.
    :return:
    """
    sent_len = len(decomp_sent)
    align_len = 0
    for span in aligned_spans:
        align_len += span[1] - span[0]

    return align_len > len_th * sent_len


def align_sub_sentence(full_sent: str,
                       decomp_sent: str,
                       tie_breaker_window: int = 3):
    """

    :param full_sent:
    :param decomp_sent:
    :return:
    """
    full_sent = full_sent.lower()
    decomp_sent = decomp_sent.lower()

    fs_toks, fs_spans = tokenize_text(full_sent)
    dc_toks, dc_spans = tokenize_text(decomp_sent)

    if len(dc_toks) <= 1 or len(fs_toks) <= 1:
        return []

    # If the last token in decomposed sentence is period, remove it
    if dc_toks[-1] == ".":
        dc_toks = dc_toks[:-1]
        dc_spans = dc_spans[:-1]

    fs_lems = lemmatize_tokens(fs_toks)
    dc_lems = lemmatize_tokens(dc_toks)

    # Find the best linear alignments between two sentences (i.e. max bipartite matching)
    # We define a simple alignment score between tokens. If two token matches, they get a score of 1.
    # To break ties, we look at nearby window of two tokens, and add a small offsets to the score based on how many
    # tokens can match.
    alignments = np.zeros((len(dc_lems), len(fs_lems)))

    for dl_idx, dl in enumerate(dc_lems):
        for fl_idx, fl in enumerate(fs_lems):
            if fl == dl:
                alignments[dl_idx, fl_idx] = 1

    # Tie-breaker offset is done by applying a 2-D convolution to in the alignment matrix
    # Construct the filter first.
    _filter = np.zeros((2 * tie_breaker_window + 1, 2 * tie_breaker_window + 1))
    _filter[tie_breaker_window, tie_breaker_window] = 1  # Reward for matching
    _filter[0:tie_breaker_window, 0:tie_breaker_window] = 0.01  # Reward for context matching
    _filter[0:tie_breaker_window, tie_breaker_window+1:] = 0.01
    _filter[tie_breaker_window+1:, 0:tie_breaker_window] = 0.01
    _filter[tie_breaker_window+1:, tie_breaker_window+1:] = 0.01

    # Apply convolution filter
    alignments = convolve2d(alignments, _filter, mode="same")

    # Max bipartite matching
    inv_alignments = np.max(alignments) - alignments
    row_ind, col_ind = linear_sum_assignment(inv_alignments)

    matched_fs_idx = []
    for dc_idx, fs_idx in zip(row_ind, col_ind):
        if alignments[dc_idx, fs_idx] >= 1:
            matched_fs_idx.append(fs_idx)

    matched_tok_spans = indices_to_span(matched_fs_idx)
    matched_char_spans = []

    for start, end in matched_tok_spans:
        matched_char_spans.append([fs_spans[start][0], fs_spans[end - 1][1]])

    return matched_char_spans


def offset_to_token_masks(proposition_spans: List[Tuple[int, int]],
                          token_spans: List[Tuple[int, int]]) -> List[int]:

    prop_token_mask = []
    for ts in token_spans:
        b_in_prop = False
        for ps in proposition_spans:
            if is_sub_span(outer_span=ps, inner_span=ts):
                b_in_prop = True
                break

        # Huggingface tokenizer set the eos token offset to be (0, 0), let's exclude that

        if ts[0] == ts[1] == 0:
            b_in_prop = False

        if b_in_prop:
            prop_token_mask.append(1)
        else:
            prop_token_mask.append(0)

    return prop_token_mask


def is_sub_span(outer_span: Tuple[int, int],
                inner_span: Tuple[int, int]) -> bool:
    """
    Check if the inner span is contained within the outer span
    :param outer_span:
    :param inner_span:
    :return:
    """
    outer_start = outer_span[0]
    inner_start = inner_span[0]
    outer_end = outer_span[1]
    inner_end = inner_span[1]

    return (outer_start <= inner_start) and (outer_end >= inner_end)


def apply_label_offset(batch_of_labels):
    """

    :param batch_of_labels: a 2d tensor of shape (N_GPUs x )
    :return:
    """
    num_rows, num_cols = batch_of_labels.size(0), batch_of_labels.size(1)
    row_ids = torch.arange(num_rows, device=batch_of_labels.device)
    row_ids = torch.unsqueeze(row_ids, 1)
    offset = row_ids.repeat(1, num_cols)
    offset = offset * num_cols
    return batch_of_labels + offset


def cosine_sim(a: List[int], b: List[int]):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def convert_retrieval_labels_to_indices(label_ids: List[List[str]], target_ids: List[str]):

    label_indices = []

    target_map = {tid: idx for idx, tid in enumerate(target_ids)}

    for labels in label_ids:
        if type(labels) is list:
            cur_lbl_indices = [target_map[l] for l in labels if l in target_map]
        else:
            cur_lbl_indices = target_map[labels] if labels in target_map else -1

        label_indices.append(cur_lbl_indices)

    return label_indices


if __name__ == '__main__':
    test_sent = "give suck to ; receive suck ."
    test_prop = "Someone is giving suck."
    spans = align_sub_sentence(test_sent, test_prop)
    print(spans)
    for start, end in spans:
        print(test_sent[start: end])
