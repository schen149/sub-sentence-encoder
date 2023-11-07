from dataclasses import dataclass, asdict
import json
import multiprocessing
from typing import List, Optional, Union, Callable, Iterable, Tuple
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from utils.label_utils import apply_label_offset, offset_to_token_masks


Span = Tuple[int, int]


@dataclass
class SentencePairExample:
    pair_id: Union[str, int]
    sent1: str
    sent2: str
    sent1_props: List[str]
    sent2_props: List[str]
    sent1_props_spans: List[List[Span]]
    sent2_props_spans: List[List[Span]]
    positive_pairs: List[List[int]]   # Index of the proposition pairs in sent1 vs sent2 that should be positive

    def asdict(self):
        return asdict(self)


@dataclass
class Proposition:
    sentence_id: str   # ID of the original sentence or document
    proposition_id: str  # ID of the proposition
    sentence_text: str
    proposition_spans: List[List[Span]]


class PropPairModelInput:
    encoder_inputs: torch.Tensor
    num_prop_per_sentence: torch.Tensor
    all_prop_mask: torch.Tensor
    prop_labels: torch.Tensor

    def __init__(self,
                 encoder_inputs,
                 num_prop_per_sentence,
                 all_prop_mask,
                 prop_labels = None):
        self.encoder_inputs = encoder_inputs
        self.num_prop_per_sentence = num_prop_per_sentence
        self.all_prop_mask = all_prop_mask
        self.prop_labels = prop_labels

    def to(self, device):
        self.encoder_inputs = self.encoder_inputs.to(device)
        self.num_prop_per_sentence = self.num_prop_per_sentence.to(device)
        self.all_prop_mask = self.all_prop_mask.to(device)
        if self.prop_labels is not None:
            self.prop_labels = self.prop_labels.to(device)

        return self


def pmap(func: Callable,
         items: Iterable,
         num_threads: int):
    """
    Applies a function on each element in a list in parallel
    :param items:
    :param func:
    :param num_threads:
    :return:
    """
    with multiprocessing.Pool(num_threads) as p:
        res = p.map(func, items)

    return res


def load_jsonl_as_dataset(dataset_path_or_dir):
    return load_dataset("json", data_files=dataset_path_or_dir)


def read_jsonl(file: str) -> List[dict]:
    examples = []
    with open(file, 'r') as fin:
        for line in fin:
            examples.append(json.loads(line))

    return examples


def read_tsv(file: str,
             fields: Optional[List[str]] = None):
    """
    Assumes first column is the fields
    :param file:
    :param fields:
    :return:
    """
    df = pd.read_csv(file, sep='\t')
    examples = df.to_dict(orient='records')
    return examples


def save_jsonl(examples, output_path):
    with open(output_path, 'w') as fout:
        for ex in examples:
            fout.write(json.dumps(ex))
            fout.write("\n")


def get_examples_with_label(all_examples, label):
    """
    Keep only exmaples with one of the labels
    :param all_examples:
    :return:
    """
    filtered = []
    for ex in all_examples:
        if ex["label"] == label:
            filtered.append(ex)

    return filtered


def gather_with_var_len(tensor,
                        base_gather_fn,
                        pad_value: int = 0,
                        add_label_offset: bool = True):
    """
    Helper for gathering 2d tensors with variable size on dim=0 from multiple GPUs/accelerator nodes.

    :param tensor: name of the (sharded) tensor that we need to gather from other accelerator nodes
    :param base_gather_fn: Base gather function from pytorch/pylightning, etc
    :return:
    """
    length = tensor.shape[0]
    all_lens = base_gather_fn(length)
    max_len = max(all_lens).item()
    if length < max_len:
        size = [max_len - length] + list(tensor.shape[1:])
        tensor = torch.cat([tensor, tensor.new_full(size, pad_value)], dim=0)
    # all gather across all processes
    data = base_gather_fn(tensor)

    # With DDP, each GPU gets its own mini-batch.
    # Since the labels are created by enumerating instances within batch, i.e. labels ranges from 0 to batch_size
    # within each batch, when merging the batches, we need to differentiate between the labels of each batch
    # This is achieved by adding a different offsets to each row of the tensor
    if add_label_offset:
        data = apply_label_offset(data)

    # delete the padding NaN items
    return torch.cat([data[i, 0:l, ...] for i, l in enumerate(all_lens)], dim=0)


def convert_example_to_features(batch_examples,
                                tokenizer,
                                max_seq_len: int,
                                sent1_key: str = "sent1",
                                sent2_key: str = None,
                                sent1_props_key: str = "sent1_props_spans",
                                sent2_props_key: str = None,
                                generate_labels: bool = False):
    batch_size = len(batch_examples)

    # Throw error if trying to generate labels for non-sentence-pair training data
    # TODO(sihaoc): maybe change this behavior in the future
    assert generate_labels == False or sent2_key is not None, "Generating labels requires sent2_key to be specified"
    
    s1_text_batch = [ex[sent1_key] for ex in batch_examples]

    if sent2_key is None:
        all_text_batch = s1_text_batch
    else:
        s2_text_batch = [ex[sent2_key] for ex in batch_examples]
        all_text_batch = s1_text_batch + s2_text_batch

    s1_spans_batch = [ex[sent1_props_key] for ex in batch_examples]

    if sent2_key is None:
        all_spans_batch = s1_spans_batch
    else:        
        s2_spans_batch = [ex[sent2_props_key] for ex in batch_examples]
        all_spans_batch = s1_spans_batch + s2_spans_batch

    all_toks_batch = tokenizer(all_text_batch,
                               return_offsets_mapping=True,
                               padding='max_length',
                               truncation=True,
                               max_length=max_seq_len)

    all_offset_mapping_batch = all_toks_batch["offset_mapping"]

    # Get proposition token masks
    all_prop_mask = []
    num_prop_per_sentence = []

    # For each sentence in a batch, convert all propositions into token mask format
    # Record the number of propositions for each sentence

    for sent_prop_spans, toks in zip(all_spans_batch, all_offset_mapping_batch):
        num_prop_per_sentence.append(len(sent_prop_spans))
        all_prop_mask += ([offset_to_token_masks(
            proposition_spans=prop_span,
            token_spans=toks
        ) for prop_span in sent_prop_spans])

    del all_toks_batch["offset_mapping"]  # Remove 'offset_mapping' from encoder input features

    # For each proposition, assign a "label" (integer) to feed into the supervised contrastive loss.
    # All the positive pairs will belong to the same
    # TODO(sihaoc): Right now, the code below doesn't correctly label the entire clique if s1_prop and s2_prop have
    #  both been assigned to different labels

    if generate_labels:
        prop_labels = np.arange(0, len(all_prop_mask))
        assigned_labels = {}

        for idx, ex in enumerate(batch_examples):
            s1_starting_offset = int(np.sum(num_prop_per_sentence[:idx]))
            s2_starting_offset = int(np.sum(num_prop_per_sentence[:batch_size + idx]))

            for s1_prop_local_id, s2_prop_local_id in ex["positive_pairs"]:
                s1_prop_global_id = s1_starting_offset + s1_prop_local_id
                s2_prop_global_id = s2_starting_offset + s2_prop_local_id

                # Check if any of the propositions have been previously assigned labels
                if s1_prop_global_id in assigned_labels:
                    gold_label = assigned_labels[s1_prop_global_id]
                elif s2_prop_global_id in assigned_labels:
                    gold_label = assigned_labels[s2_prop_global_id]
                else:
                    gold_label = s1_prop_global_id

                # make assignments
                prop_labels[s1_prop_global_id] = gold_label
                prop_labels[s2_prop_global_id] = gold_label

        prop_labels = torch.tensor(prop_labels)

    else:
        prop_labels = None

    # Convert everything into tensors
    encoder_input_batch = all_toks_batch.convert_to_tensors(tensor_type='pt')
    num_prop_per_sentence = torch.tensor(num_prop_per_sentence, dtype=torch.int32)
    all_prop_mask = torch.tensor(all_prop_mask, dtype=torch.float)

    return PropPairModelInput(encoder_input_batch, num_prop_per_sentence, all_prop_mask, prop_labels)


def convert_propositions_to_features(batch_props,
                                     tokenizer,
                                     max_seq_len: int):

    all_text_batch = [ex["sentence_text"] for ex in batch_props]
    all_spans_batch = [ex["spans"] for ex in batch_props]
    all_toks_batch = tokenizer(all_text_batch,
                               return_offsets_mapping=True,
                               padding='max_length',
                               truncation=True,
                               max_length=max_seq_len)

    all_offset_mapping_batch = all_toks_batch["offset_mapping"]


    # Get proposition token masks
    all_prop_mask = []
    num_prop_per_sentence = [1] * len(all_text_batch)  # each example only has 1 proposition

    for sent_prop_spans, toks in zip(all_spans_batch, all_offset_mapping_batch):
        prop_span_formatted = []
        for prop_span in sent_prop_spans:
            # TODO(sihaoc): legacy code has each span as {"start":..., "end": ...} instead of [int, int] tuple
            if type(prop_span) == dict:
                prop_span = [prop_span["start"], prop_span["end"]]

            prop_span_formatted.append(prop_span)

        all_prop_mask.append(offset_to_token_masks(
            proposition_spans=prop_span_formatted,
            token_spans=toks
        ))

    del all_toks_batch["offset_mapping"]  # Remove 'offset_mapping' from encoder input features

    # Convert everything into tensors
    encoder_input_batch = all_toks_batch.convert_to_tensors(tensor_type='pt')
    num_prop_per_sentence = torch.tensor(num_prop_per_sentence, dtype=torch.int32)
    all_prop_mask = torch.tensor(all_prop_mask, dtype=torch.float)

    return PropPairModelInput(
        encoder_input_batch,
        num_prop_per_sentence,
        all_prop_mask,
        None  # No label
    )


def convert_propositions_to_sent_encoder_inputs(batch_props,
                                                tokenizer,
                                                max_seq_len: int):
    prop_model_inputs = convert_propositions_to_features(batch_props, tokenizer, max_seq_len)
    enc_inputs = prop_model_inputs.encoder_inputs
    prop_attn_masks = prop_model_inputs.all_prop_mask

    enc_inputs["pooler_mask"] = prop_attn_masks
    return enc_inputs


def convert_sentence_pair_encoded_outputs(encoded, num_prop_per_sent):
    """
    Convert encoded outputs to match all sentence inputs
    :param encoded:
    :param batch_input:
    :return:
    """
    encoded = encoded.to("cpu").detach()
    num_prop_per_sent = num_prop_per_sent.to("cpu")
    num_prop_per_sent = num_prop_per_sent.view(2, -1)
    encoded_list = []
    for i in range(num_prop_per_sent.size(1)):
        sent1_row_start = torch.sum(num_prop_per_sent[0, :i])
        sent1_row_end = torch.sum(num_prop_per_sent[0, :i + 1])

        sent1_total_offset = torch.sum(num_prop_per_sent[0])

        sent2_row_start = sent1_total_offset + torch.sum(num_prop_per_sent[1, :i])
        sent2_row_end = sent1_total_offset + torch.sum(num_prop_per_sent[1, :i + 1])

        sent1_encoded = encoded[sent1_row_start:sent1_row_end]
        sent2_encoded = encoded[sent2_row_start:sent2_row_end]
        sent1_encoded = sent1_encoded.tolist()
        sent2_encoded = sent2_encoded.tolist()
        encoded_list.append({
            "sent1_props_encoded": sent1_encoded,
            "sent2_props_encoded": sent2_encoded
        })

    return encoded_list


if __name__ == '__main__':
    pass
