import argparse, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

p=argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--loops", type=int, default=10)
args=p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(
    args.model_dir, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True
).eval()

# 构造同构 prompt（LLM 吞吐压测不看内容）
msgs=[{"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":"Generate a succinct trading heuristic idea in one sentence."}]
prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

enc = tok([prompt]*args.batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
enc = {k:v.to(mdl.device) for k,v in enc.items()}
# 预热
mdl.generate(**enc, max_new_tokens=8, do_sample=False)

torch.cuda.synchronize()
t0=time.time()
tokens_out=0
for _ in range(args.loops):
    out = mdl.generate(**enc, max_new_tokens=args.tokens, do_sample=False,
                       eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    tokens_out += args.batch*args.tokens
torch.cuda.synchronize()
t1=time.time()

print(f"Device: {mdl.device}")
print(f"Batch={args.batch}, tokens/gen={args.tokens}, loops={args.loops}")
print(f"Total new tokens: {tokens_out}, elapsed: {t1-t0:.2f}s, throughput: {tokens_out/(t1-t0):.1f} toks/s")
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak CUDA mem: {mem:.1f} GB")
