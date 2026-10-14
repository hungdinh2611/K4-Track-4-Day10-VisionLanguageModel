"""Kaggle T4 workflow for the Day 10 Qwen3-VL road damage lab.

Run from the repository root. Dataset credentials are read from environment;
the script never writes credentials or model weights into tracked locations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = ("Detect road-surface damage in this image. Return only a JSON list. "
          "Each item must have label pothole, crack, or manhole and bbox_2d "
          "[x1,y1,x2,y2] as integers on the 0-1000 normalized scale. "
          "Use [] when there are no objects.")
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
LLM_RE = r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
VIT_RE = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    scale_x, scale_y = (1000, 1000) if coord == "norm1000" else (width, height)
    vals = ((xc-w/2)*scale_x, (yc-h/2)*scale_y,
            (xc+w/2)*scale_x, (yc+h/2)*scale_y)
    return {"label": label, "bbox_2d": [max(0, min(scale_x if i % 2 == 0 else scale_y, round(v)))
                                          for i, v in enumerate(vals)]}


def target_json(record, coord="norm1000"):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, record["width"], record["height"])
             for o in record["objects"]]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def show_split_examples(n=6):
    import matplotlib.pyplot as plt
    rows = read_jsonl("data/splits/train.jsonl")[:n]
    fig, axes = plt.subplots(2, math.ceil(len(rows)/2), figsize=(16, 8))
    for ax, row in zip(axes.flat, rows):
        im = Image.open(Path("data") / row["image_path"]).convert("RGB")
        draw = ImageDraw.Draw(im)
        for obj in row["objects"]:
            xc, yc, w, h = obj["yolo"]
            box = ((xc-w/2)*im.width, (yc-h/2)*im.height,
                   (xc+w/2)*im.width, (yc+h/2)*im.height)
            draw.rectangle(box, outline="red", width=2); draw.text((box[0], box[1]), obj["label"], fill="yellow")
        ax.imshow(im); ax.axis("off")
    for ax in axes.flat[len(rows):]: ax.axis("off")
    plt.tight_layout(); plt.show()


def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import prepare_model_for_kbit_training
    qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=qconfig, dtype=torch.float16,
        device_map={"": 0}, low_cpu_mem_usage=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, processor


def make_messages(image, prompt=PROMPT):
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                           {"type": "text", "text": prompt}]}]


def run_inference(model, processor, split="test", adapter=None, out=None, limit=None):
    from peft import PeftModel
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        extra = Path(adapter) / "extra_trainable.pt"
        if extra.exists():
            state = torch.load(extra, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
    rows = read_jsonl(f"data/splits/{split}.jsonl")
    if limit: rows = rows[:limit]
    outputs = []
    model.eval()
    for row in rows:
        image = Image.open(Path("data") / row["image_path"]).convert("RGB")
        inputs = processor.apply_chat_template(make_messages(image), tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        raw = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0]
        outputs.append({"image_id": row["image_id"], "raw": raw})
    out = Path(out or f"work/{split}.jsonl")
    write_jsonl(out, outputs)
    del model; torch.cuda.empty_cache()
    return out


class RoadDataset(Dataset):
    def __init__(self, rows, processor, coord="norm1000"):
        self.rows, self.processor, self.coord = rows, processor, coord
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(Path("data") / row["image_path"]).convert("RGB")
        user = make_messages(image)
        prompt = self.processor.apply_chat_template(user, tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        full = self.processor.apply_chat_template(user + [{"role": "assistant", "content": target_json(row, self.coord)}],
            tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
        n = prompt["input_ids"].shape[1]
        if not torch.equal(prompt["input_ids"][0], full["input_ids"][0, :n]):
            raise ValueError("Chat template prompt is not a prefix of the full conversation")
        labels = full["input_ids"].clone(); labels[:, :n] = -100
        labels[full["attention_mask"] == 0] = -100
        return {**full, "labels": labels, "image_id": row["image_id"]}


def collate(examples, processor):
    pad_id = processor.tokenizer.pad_token_id
    maxlen = max(x["input_ids"].shape[1] for x in examples)
    result = {}
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        if key not in examples[0]: continue
        pad = -100 if key == "labels" else (pad_id if key == "input_ids" else 0)
        vals = [torch.nn.functional.pad(x[key], (0, maxlen-x[key].shape[1]), value=pad) for x in examples]
        result[key] = torch.cat(vals, dim=0)
    for key in ("pixel_values", "image_grid_thw"):
        if key in examples[0]: result[key] = torch.cat([x[key] for x in examples], dim=0)
    return result


def configure_lora(model, experiment="e1", rank=16):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters(): p.requires_grad_(False)
    model.enable_input_require_grads()
    targets = LLM_RE if experiment != "e3" else f"({LLM_RE})|({VIT_RE})"
    peft_model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2*rank,
        lora_dropout=0.05, target_modules=targets, modules_to_save=None))
    if experiment in ("e2", "e3"):
        for name, p in peft_model.named_parameters():
            if any(key in name for key in PROJECTOR_KEYS):
                p.data = p.data.float(); p.requires_grad_(True)
    # The CUDA GradScaler refuses FP16 gradients during unscale_. Keep every
    # trainable adapter/projector weight in FP32; autocast still runs base ops
    # in FP16 and the optimizer state remains numerically stable.
    for p in peft_model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    peft_model.config.use_cache = False
    return peft_model


def train(experiment="e1", rank=16, max_steps=300, data_fraction=1.0):
    if not torch.cuda.is_available(): raise RuntimeError("Select a Kaggle GPU accelerator (T4 x2 or P100)")
    seed = 42; random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model, processor = load_model()
    model = configure_lora(model, experiment, rank)
    rows = read_jsonl("data/splits/train.jsonl")
    random.Random(seed).shuffle(rows)
    rows = rows[:max(1, int(len(rows)*data_fraction))]
    dataset = RoadDataset(rows, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=lambda xs: collate(xs, processor), num_workers=0)
    trainable = [p for p in model.parameters() if p.requires_grad]
    bad_dtypes = {str(p.dtype) for p in trainable if p.dtype != torch.float32}
    if bad_dtypes:
        raise TypeError(f"trainable parameters must be fp32 for GradScaler; found {sorted(bad_dtypes)}")
    count = sum(p.numel() for p in trainable)
    # Use FP32 master weights. Some multimodal modules produce FP16 gradients
    # even when their trainable leaf weights are FP32; keeping Adam's state and
    # gradients on master copies avoids mixed-dtype foreach optimizer failures.
    master_params = [torch.nn.Parameter(p.detach().float().clone()) for p in trainable]
    opt = torch.optim.AdamW(master_params, lr=2e-4, weight_decay=0, foreach=False)
    steps_warm = max(1, int(max_steps*.05)); scaler = torch.amp.GradScaler("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(); opt.zero_grad(set_to_none=True); step = 0; micro = 0; t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    run_dir = Path(f"work/{experiment}_r{rank}"); run_dir.mkdir(parents=True, exist_ok=True)
    while step < max_steps:
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16): loss = model(**batch).loss
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            scaler.scale(loss / 4).backward(); micro += 1
            if micro < 4: continue
            promoted = 0
            for p, master in zip(trainable, master_params):
                if p.grad is not None:
                    promoted += int(p.grad.dtype == torch.float16)
                    master.grad = p.grad.detach().float().clone()
            if promoted:
                print(f"copied {promoted} FP16 gradients to FP32 master weights")
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(master_params, 1.0)
            lr_scale = min(1.0, (step+1)/steps_warm) if step < steps_warm else .5*(1+math.cos(math.pi*(step-steps_warm)/max(1, max_steps-steps_warm)))
            for group in opt.param_groups: group["lr"] = 2e-4*lr_scale
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                for p, master in zip(trainable, master_params):
                    p.copy_(master.to(dtype=p.dtype))
            opt.zero_grad(set_to_none=True)
            for p in trainable: p.grad = None
            step += 1; micro = 0
            if step % 25 == 0: print(f"{experiment} step={step}/{max_steps} loss={loss.item():.4f}")
            if step % 100 == 0:
                checkpoint = run_dir / f"checkpoint-{step}"
                model.save_pretrained(checkpoint)
                projector = {n: p.detach().float().cpu() for n, p in model.named_parameters()
                             if any(k in n for k in PROJECTOR_KEYS) and p.requires_grad}
                if projector: torch.save(projector, checkpoint / "extra_trainable.pt")
            if step >= max_steps: break
    minutes = (time.time()-t0)/60
    model.save_pretrained(run_dir)
    extra = {n: p.detach().float().cpu() for n,p in model.named_parameters()
             if any(k in n for k in PROJECTOR_KEYS) and p.requires_grad}
    if extra: torch.save(extra, run_dir / "extra_trainable.pt")
    meta = {"experiment": experiment, "rank": rank, "trainable_params": count,
            "peak_vram_gb": torch.cuda.max_memory_allocated()/1024**3,
            "train_minutes": minutes, "steps": step}
    (run_dir/"run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2)); return run_dir, meta


def score(pred, name, meta=None, coord="norm1000", split="test", evidence_path=None):
    import subprocess, sys
    args = [sys.executable, "scripts/score.py", "--pred", str(pred), "--name", name,
            "--out", str(evidence_path or f"report/evidence/{name.lower()}.json"),
            "--split", split, "--coord", coord]
    for k,v in (meta or {}).items(): args += ["--meta", f"{k}={v}"]
    subprocess.run(args, check=True)


def evaluate_adapter(adapter_dir, name, meta, evidence_path=None):
    model, processor = load_model()
    prediction_path = run_inference(model, processor, "test", str(adapter_dir),
                                    f"work/{name.lower()}_test.jsonl")
    del model, processor
    torch.cuda.empty_cache()
    score(prediction_path, name, meta, evidence_path=evidence_path)
    return prediction_path


def run_all(steps=300):
    """Run the required zero-shot, SFT ablations, and rank ablation end-to-end."""
    Path("report/evidence").mkdir(parents=True, exist_ok=True)
    Path("work").mkdir(exist_ok=True)
    if not Path("report/evidence/e0.json").exists():
        print("[E0] zero-shot inference")
        model, processor = load_model()
        p0 = run_inference(model, processor, "test", out="work/e0_test.jsonl")
        del model, processor
        torch.cuda.empty_cache()
        score(p0, "E0")
    else:
        print("[E0] reusing completed evidence")

    best = None
    for experiment in ("e1", "e2", "e3"):
        existing = Path(f"report/evidence/{experiment}.json")
        if experiment == "e1" and existing.exists():
            prior = json.loads(existing.read_text(encoding="utf-8"))
            if prior.get("score_version") == 1 and prior.get("split") == "test":
                print("[E1] reusing completed evidence")
                best = (experiment, None, prior.get("meta", {}))
                continue
        print(f"[{experiment.upper()}] training")
        adapter, meta = train(experiment=experiment, rank=16, max_steps=steps)
        evaluate_adapter(adapter, experiment.upper(), meta)
        if best is None:
            best = (experiment, adapter, meta)

    # E4 changes only LoRA rank relative to E1 (r=16 baseline).
    for rank in (4, 64):
        print(f"[E4 rank={rank}] training")
        adapter, meta = train(experiment="e1", rank=rank, max_steps=steps)
        evidence = f"report/evidence/e4_rank_{rank}.json"
        evaluate_adapter(adapter, f"E4-rank{rank}", meta, evidence_path=evidence)

    print("All requested experiments finished. Evidence is in report/evidence/.")
    print("E1 baseline metadata:", json.dumps(best[2], indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "e0", "train", "infer", "score", "all"])
    ap.add_argument("--experiment", default="e1", choices=["e1","e2","e3","e4"])
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--adapter"); ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--pred"); ap.add_argument("--name", default="E0")
    a = ap.parse_args()
    if a.command == "all":
        run_all(a.steps)
    elif a.command == "prepare":
        import subprocess, sys
        subprocess.run([sys.executable,"scripts/prepare_data.py","--kaggle"], check=True)
    elif a.command in ("e0", "infer"):
        m,p = load_model(); out = run_inference(m,p,a.split,a.adapter, f"work/{a.name.lower()}.jsonl")
        print(out)
    elif a.command == "train":
        train(a.experiment, a.rank, a.steps)
    elif a.command == "score": score(a.pred, a.name)

if __name__ == "__main__": main()
