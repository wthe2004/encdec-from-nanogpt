# generate.py
import torch
from tokenizers import Tokenizer
import tyro

# 从我们自己的模块中导入
from config import GenerateArgs
from model import TransformerDecoder


if __name__ == "__main__":
    args = tyro.cli(GenerateArgs)

    torch.manual_seed(args.seed)

    # --- 加载Tokenizer (直接从文件加载) ---
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # --- 加载模型 ---
    model = TransformerDecoder(vocab_size, args.model).to(args.device)
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    # --- 生成文本 ---
    print(f"\n--- Generating text from prompt: '{args.prompt}' ---")
    prompt_ids = tokenizer.encode(args.prompt).ids
    input_tensor = torch.tensor(
        prompt_ids, dtype=torch.long, device=args.device
    ).unsqueeze(0)
    generated_ids = model.generate(input_tensor, args.num_tokens_to_generate)
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(), skip_special_tokens=True
    )

    print(generated_text)
    print("\n--------------------")
