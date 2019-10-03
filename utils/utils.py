import torch


def sequence_mask(sequence_length, maxlen=None):
    if maxlen is None:
        maxlen = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, maxlen).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, maxlen)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.to(sequence_length.device)
    seq_length_expand = (sequence_length.unsqueeze(1).
                         expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
