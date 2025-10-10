# config.py
from dataclasses import dataclass, field


@dataclass
class ModelArgs:
    """
    Arguments related to the model architecture.
    """

    block_size: int = 128
    n_embd: int = 384
    n_layers: int = 6
    num_heads: int = 6  # 注意：我们把默认头数统一为6
    dropout: float = 0.0


@dataclass
class TrainArgs:
    """
    Arguments for the training task.
    """

    # --- overfitting test ---
    overfit_test: bool = False

    # --- Tokenizer ---
    vocab_size: int = 16000

    # --- 文件与路径 ---
    dataset_name: str = "IWSLT/da2023"
    lp: str = "en-de"  # language-pair
    tokenizer_path: str = f"outputs/bpe_iwslt2023_vocab{vocab_size}.json"
    exp_name: str = f"iwslt2023_seq2seq"
    split_name: str = "train"
    out_dir: str = "outputs"

    # --- 训练过程 ---
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 1337
    learning_rate: float = 3e-4
    max_iters: int = 50
    eval_interval: int = 100
    eval_iters: int = 200
    batch_size: int = 32
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    max_epochs: int = 8

    # --- 模型架构 (嵌套的dataclass) ---
    model: ModelArgs = field(default_factory=ModelArgs)


@dataclass
class GenerateArgs:
    """
    Arguments for the generation task.
    """

    # --- 文件与路径 ---
    model_path: str

    # --- 生成过程 ---
    prompt: str = "Hello, world!\n"
    max_new_tokens: int = 500
    temperature: float = 0.0001
    top_k: int = 200
    top_p: float = 0.95
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    seed: int = 1337

    # --- 模型架构 (必须与加载的模型匹配) ---
    model: ModelArgs = field(default_factory=ModelArgs)
