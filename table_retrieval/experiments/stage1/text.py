"""Prepares frozen Stage 1 table inputs for offline index construction."""
from tqdm.auto import tqdm

from ...data import check_id
from ...stage1.text import table_text


def prepare_text_inputs(corpus, tokenizer, max_length):
    """Tokenize once, recording tables truncated to the configured index budget."""
    ids = sorted(check_id(uid) for uid in corpus)
    encoded, clipped = [], []
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    for uid in tqdm(ids, desc='Preparing shared table text'):
        text = table_text(corpus[uid])
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
        if len(token_ids) > budget:
            clipped.append({'table_id': uid, 'original_tokens': len(token_ids)})
        encoded.append(tokenizer(text, add_special_tokens=True, truncation=True,
                                 max_length=max_length)['input_ids'])
    return ids, encoded, clipped
