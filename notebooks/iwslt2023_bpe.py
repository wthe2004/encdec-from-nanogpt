from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


def batch_iterator(dataset, batch_size=1000):
    for i in range(0, len(dataset["train"]), batch_size):
        batch = dataset["train"][i : i + batch_size]
        yield [text for text in batch["src"] if text is not None]
        yield [text for text in batch["ref"] if text is not None]


if __name__ == "__main__":
    dataset = load_dataset("IWSLT/da2023")
    print(dataset)
    en_de_dataset = dataset.filter(lambda x: x["lp"] == "en-de")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=30000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )

    tokenizer.train_from_iterator(
        batch_iterator(dataset=en_de_dataset), trainer=trainer
    )

    tokenizer.save("iwslt-da2023-bpe-tokenizer.json")
