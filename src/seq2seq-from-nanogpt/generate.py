# generate.py
import torch
import tyro
from tokenizers import Tokenizer

# 从我们自己的模块中导入
from config import GenerateArgs, ModelArgs  # 导入 ModelArgs 以便类型提示
from model import Seq2SeqTransformer


def generate(args: GenerateArgs):
    """
    加载一个自包含的 checkpoint (模型+分词器) 并生成翻译文本。
    """
    print("Starting generation...")
    torch.manual_seed(args.seed)

    # 1. 加载自包含的 Checkpoint
    try:
        checkpoint = torch.load(
            args.model_path, map_location=args.device, weights_only=False
        )
        print(f"Self-contained checkpoint loaded from: {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    # 2. 从 Checkpoint 恢复分词器
    tokenizer_str = checkpoint.get("tokenizer_str")
    if tokenizer_str is None:
        print("Error: 'tokenizer_str' not found in the checkpoint.")
        return
    tokenizer = Tokenizer.from_str(tokenizer_str)
    print("Tokenizer restored from checkpoint.")

    # 3. 从 Checkpoint 中恢复模型架构参数
    model_args: ModelArgs = checkpoint.get("model_args")
    if model_args is None:
        print("Error: 'model_args' not found in the checkpoint.")
        return
    print(f"Model arguments restored from checkpoint: {model_args}")

    # 4. 初始化模型
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    sos_token_id = tokenizer.token_to_id("[CLS]")
    eos_token_id = tokenizer.token_to_id("[SEP]")

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        args=model_args,
        pad_token_id=pad_token_id,
        sos_token_id=sos_token_id,
        eos_token_id=eos_token_id,
    )
    model.to(args.device)

    # 5. 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model initialized and weights loaded successfully.")

    # 6. 准备输入
    print(f"\nSource sentence: '{args.prompt}'")
    src_tokens = tokenizer.encode(args.prompt).ids
    src_tensor = torch.tensor(
        src_tokens, dtype=torch.long, device=args.device
    ).unsqueeze(0)

    # 7. 执行生成
    with torch.no_grad():
        generated_tokens = model.generate(
            src=src_tensor,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
        )

    # 8. 解码并打印结果
    generated_text = tokenizer.decode(generated_tokens[0].tolist())

    print("-" * 30)
    print(f"Generated translation: '{generated_text}'")
    print("-" * 30)


if __name__ == "__main__":
    args = tyro.cli(GenerateArgs)
    generate(args)
