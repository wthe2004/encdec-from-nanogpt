import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset["train"]), batch_size):
        batch = dataset["train"][i : i + batch_size]
        yield [text for text in batch["src"] if text is not None]
        yield [text for text in batch["ref"] if text is not None]


def get_bpe_tokenizer(
    dataset_name: str,
    vocab_size: int,
    tokenizer_path: str,
    lp: str = "en-de",
) -> Tokenizer:
    """
    Trains a new BPE tokenizer or loads an existing one.
    """
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)
    else:
        return train_bpe_tokenizer(dataset_name, vocab_size, tokenizer_path, lp=lp)


def train_bpe_tokenizer(
    dataset_name: str,
    vocab_size: int,
    tokenizer_path: str,
    lp="en-de",
) -> Tokenizer:
    raw_dataset = load_dataset(dataset_name)
    dataset = raw_dataset.filter(lambda x: x["lp"] == lp)
    print("Training a new BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )

    tokenizer.train_from_iterator(batch_iterator(dataset=dataset), trainer=trainer)

    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to: {tokenizer_path}")
    return tokenizer


if __name__ == "__main__":

    tokenizer = get_bpe_tokenizer(
        dataset_name="IWSLT/da2023",
        vocab_size=300,
        tokenizer_path="bpe_iwslt2023_vocab300.json",
        lp="en-de",
    )

    print(tokenizer)
