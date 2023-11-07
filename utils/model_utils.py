from model.pl_subencoder import LitSubEncoder
from transformers import T5ForConditionalGeneration, AutoConfig, AutoModel
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def load_seq2seq_model_from_deepspeed(model_name,
                                      ckpt_path):
    """
    Load deepspeed model weights
    """
    cfg = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        hf_model = T5ForConditionalGeneration._from_config(cfg)

    hf_model = load_checkpoint_and_dispatch(hf_model, ckpt_path, device_map='auto')

    return hf_model

def load_model(model_path):
    return LitSubEncoder.load_from_checkpoint(model_path)


def load_huggingface_model(model_name):
    return AutoModel.from_pretrained(model_name)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        print(name, params)
        total_params += params

    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':
    model_name = "sentence-transformers/sentnece-t5-base"
    model = AutoModel.from_pretrained(model_name)
    count_parameters(model)
