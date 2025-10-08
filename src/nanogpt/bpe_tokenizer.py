import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def get_bpe_tokenizer(
    train_file: str,
    vocab_size: int,
    tokenizer_path: str,
) -> Tokenizer:
    """
    Trains a new BPE tokenizer or loads an existing one.
    """
    if os.path.exists(tokenizer_path):
        print(f"Loading existing tokenizer: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)
    else:
        return train_bpe_tokenizer(train_file, vocab_size, tokenizer_path)


def train_bpe_tokenizer(
    train_file: str, vocab_size: int, tokenizer_path: str
) -> Tokenizer:
    print("Training a new BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train([train_file], trainer)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to: {tokenizer_path}")
    return tokenizer


if __name__ == "__main__":
    # Example usage
    vocab_size = 5000
    train_file = "./shakespeare.txt"
    tokenizer_path = f"bpe_shakespeare_vocab{vocab_size}.json"
    tokenizer = train_bpe_tokenizer(train_file, vocab_size, tokenizer_path)

    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    # save
    tokenizer.save(tokenizer_path)
