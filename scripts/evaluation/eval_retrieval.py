"""
Evaluation with sub-encoders for retrieval tasks.
"""
from functools import partial

from utils.model_utils import load_model, load_huggingface_model
from utils.data_utils import read_jsonl, convert_propositions_to_features, convert_propositions_to_sent_encoder_inputs
from utils.label_utils import convert_retrieval_labels_to_indices

from model.layers import Pooling

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import os

from typing import Iterable

def encode_with_subencoder(
        propositions,
        model,
        tokenizer,
        batch_size,
        max_seq_len,
        cuda
):
    collate_fn = partial(
        convert_propositions_to_features,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    dataloader = DataLoader(propositions,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False)

    all_encoded = []
    for batch in tqdm(dataloader):
        if cuda:
            batch.to("cuda")

        with torch.no_grad():
            props_encoded = model(
                encoder_inputs=batch.encoder_inputs,
                num_prop_per_sentence=batch.num_prop_per_sentence,
                all_prop_mask=batch.all_prop_mask
            )

        props_encoded = props_encoded.detach().cpu()
        all_encoded.append(props_encoded)

    all_encoded = torch.cat(all_encoded, dim=0)

    return all_encoded

def encode_with_sent_encoder(propositions,
                             encode_fn,
                             tokenizer,
                             batch_size,
                             max_seq_len,
                             cuda):

    collate_fn = partial(
        convert_propositions_to_sent_encoder_inputs,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    dataloader = DataLoader(propositions,
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False)
    all_encoded = []
    pooler = Pooling()
    for batch in tqdm(dataloader):
        if cuda:
            batch.to("cuda")

        pooler_mask = batch["pooler_mask"]
        del batch["pooler_mask"]

        with torch.no_grad():
            props_encoded = encode_fn(**batch)
            last_hidden_state = props_encoded["last_hidden_state"]
            prop_repr = pooler(last_hidden_state, pooler_mask)

        prop_repr = prop_repr.detach().cpu()
        all_encoded.append(prop_repr)

    all_encoded = torch.cat(all_encoded, dim=0)

    return all_encoded


def convert_prop_id_to_doc(prop_id):
    return "_".join(prop_id.split("_")[:2])


def convert_prop_id_to_sent(prop_id):
    return "_".join(prop_id.split("_")[:3])


def convert_prop_id_to_cluster(prop_id):
    return "_".join(prop_id.split("_")[:1])

def dedup_ranks(rank_ids):
    seen_ids = set()
    res = []
    for id in rank_ids:
        if id not in seen_ids:
            res.append(id)
            seen_ids.add(id)
    
    return res

def evaluate_propsegment(queries_encoded,
                         targets_encoded,
                         query_ids,
                         query_labels,
                         target_ids,
                         topk: Iterable[int] = (1, 5, 10, 20),
                         cuda: bool = True):

    # Given the target ids 
    query_id_indices = convert_retrieval_labels_to_indices(query_ids, target_ids)
    label_indices = convert_retrieval_labels_to_indices(query_labels, target_ids)

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    topk = sorted(topk)
    maxk = topk[-1] + 1

    if cuda:
        targets_encoded = targets_encoded.to("cuda")

    topk_hits = {k: 0 for k in topk}
    topk_sent_hits = {k: 0 for k in topk}
    topk_doc_hits = {k: 0 for k in topk}
    topk_cluster_hits = {k: 0 for k in topk}

    for idx, q in enumerate(tqdm(queries_encoded)):
        if cuda:
            q = q.to("cuda")

        q = q.expand(targets_encoded.size(0), -1)
        sim = cos_sim(q, targets_encoded)
        rank = torch.argsort(sim, descending=True).cpu().numpy()[:maxk].tolist()
        cur_qid = query_id_indices[idx]
        cur_labels = label_indices[idx]

        if cur_qid in rank:
            rank.remove(cur_qid)  # remove self-retrieval

        for k in topk:
            cands = rank[:k]
            b_hit = 0
            for lbl in cur_labels:
                if lbl in cands:
                    b_hit += 1

            # Sum up the recall and we will report the average
            if k == 1:
                topk_hits[k] += b_hit
            else:
                topk_hits[k] += b_hit / len(cur_labels)


        # Additionally, convert the labels to sentence & paragraph level

        rank_target_ids = [target_ids[r] for r in rank]
        rank_label_id = query_labels[idx]
        rank_target_sent_ids = [convert_prop_id_to_sent(i) for i in rank_target_ids]
        rank_label_sent_id = list(set([convert_prop_id_to_sent(i) for i in rank_label_id]))

        rank_target_doc_ids = [convert_prop_id_to_doc(i) for i in rank_target_ids]
        rank_label_doc_id = list(set([convert_prop_id_to_doc(i) for i in rank_label_id]))

        rank_target_cluster_ids = [convert_prop_id_to_cluster(i) for i in rank_target_ids]
        rank_label_cluster_id = list(set([convert_prop_id_to_cluster(i) for i in rank_label_id]))
        
        # Deduplicate the rank, so that recall @ k becomes a fair comparison
        rank_target_sent_ids = dedup_ranks(rank_target_sent_ids)
        rank_target_doc_ids = dedup_ranks(rank_target_doc_ids)
        rank_target_cluster_ids = dedup_ranks(rank_target_cluster_ids)

        for k in topk:
            sent_cands = rank_target_sent_ids[:k]
            doc_cands = rank_target_doc_ids[:k]
            cluster_cands = rank_target_cluster_ids[:k]

            b_sent_hit = 0
            b_doc_hit = 0
            b_cluster_hit = 0
            for lbl in rank_label_sent_id:
                if lbl in sent_cands:
                    b_sent_hit += 1

            for lbl in rank_label_doc_id:
                if lbl in doc_cands:
                    b_doc_hit += 1

            for lbl in rank_label_cluster_id:
                if lbl in cluster_cands:
                    b_cluster_hit += 1

            # Sum up the recall values
            if k == 1:
                topk_sent_hits[k] += b_sent_hit
                topk_doc_hits[k] += b_doc_hit
                topk_cluster_hits[k] += b_cluster_hit
            else:
                topk_sent_hits[k] += b_sent_hit / len(rank_label_sent_id)
                topk_doc_hits[k] += b_doc_hit / len(rank_label_doc_id)
                topk_cluster_hits[k] += b_cluster_hit / len(rank_label_cluster_id)

    total_n = len(query_id_indices)
    for k, nhits in topk_hits.items():
        m_prefix = "Success" if k == 1 else "Recall"
        print(f"{m_prefix} @ {k} = {nhits} / {total_n} = {nhits / total_n:.4f}")

    print("Sentence-Level retrieval...")
    for k, nhits in topk_sent_hits.items():
        m_prefix = "Success" if k == 1 else "Recall"
        print(f"\t{m_prefix} @ {k} = {nhits} / {total_n} = {nhits / total_n:.4f}")

    print("Paragraph-Level retrieval...")
    for k, nhits in topk_doc_hits.items():
        m_prefix = "Success" if k == 1 else "Recall"
        print(f"\t{m_prefix} @ {k} = {nhits} / {total_n} = {nhits / total_n:.4f}")

    print("Cluster-Level retrieval...")
    for k, nhits in topk_cluster_hits.items():
        m_prefix = "Success" if k == 1 else "Recall"
        print(f"\t{m_prefix} @ {k} = {nhits} / {total_n} = {nhits / total_n:.4f}")


def save_encoded(encoded, output_path):
    """
    Save the encoded outputs to a file
    :param examples: a list of N dicts containing the input propositions
    :param encoded: a N x d tensor containing encoded propositions
    :return:
    """
    f = open(output_path, 'wb')
    torch.save(encoded, f)


def load_encoded(path):
    f = open(path, 'rb')
    return torch.load(f)


def main(args):
    # load model
    if args.baseline_name is not None:
        model = load_huggingface_model(args.baseline_name)
        tokenizer = AutoTokenizer.from_pretrained(args.baseline_name)

    else:
        model = load_model(args.checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(model.params.model_name)

    if args.cuda:
        model = model.to("cuda")

    queries = read_jsonl(args.query_path)
    targets = read_jsonl(args.target_path)

    print("Constructing indices from labels ids...")
    query_ids = [q["id"] for q in queries]
    query_labels = [q["label"] for q in queries]
    target_ids = [t["id"] for t in targets]

    using_cache = args.use_cache and os.path.exists(args.query_output_path) and os.path.exists(args.target_output_path)
    if using_cache:
        print(f"Loading encoded propositions from {args.query_output_path} and {args.target_output_path}")
        queries_encoded = load_encoded(args.query_output_path)
        targets_encoded = load_encoded(args.target_output_path)
    else:
        print("Encoding Propositions")
        if args.baseline_name is not None:
            if "t5" in args.baseline_name:
                encode_fn = model.encoder.forward
            else:
                encode_fn = model.forward

            prop_encode_fn = partial(
                encode_with_sent_encoder,
                encode_fn=encode_fn,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                cuda=args.cuda)
        else:
            prop_encode_fn = partial(
                encode_with_subencoder,
                model=model,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                cuda=args.cuda)

        queries_encoded = prop_encode_fn(propositions=queries)
        targets_encoded = prop_encode_fn(propositions=targets)

    if not using_cache:
        print(f"Saving the encoded queries and targets to {args.query_output_path} and {args.target_output_path}")

        # Save the encoded results
        if args.query_output_path:
            save_encoded(queries_encoded, args.query_output_path)

        if args.target_output_path:
            save_encoded(targets_encoded, args.target_output_path)

    print(f"Running evaluation...")

    evaluate_propsegment(queries_encoded, targets_encoded, query_ids, query_labels, target_ids)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_name",
        help="baseline sentence encoder to use",
        default=None,
        type=str
    )
    parser.add_argument(
        "--checkpoint_path",
        help="path to the lightning checkpoint",
        default=None,
        type=str
    )
    parser.add_argument(
        "--query_path",
        help="path to file containing queries",
        type=str
    )
    parser.add_argument(
        "--target_path",
        help="path to file containing the targets",
        type=str
    )
    parser.add_argument(
        "--query_output_path",
        help="path to results",
        default=None,
        type=str
    )
    parser.add_argument(
        "--target_output_path",
        help="path to results",
        default=None,
        type=str
    )
    parser.add_argument(
        "--batch_size",
        help="batch size during evaluation",
        default=32,
        type=int
    )
    parser.add_argument(
        "--max_seq_len",
        help="max seuqnece length during tokenization",
        default=128,
        type=int
    )
    parser.add_argument(
        "--cuda",
        help="max seuqnece length during tokenization",
        action="store_true"
    )
    parser.add_argument(
        "--use_cache",
        help="use cached encoded queries and targets for eval",
        action="store_true"
    )
    args = parser.parse_args()

    main(args)