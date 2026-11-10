# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "transformers==5.22.2",
#     "trl==1.12.0",
#     "peft",
#     "accelerate",
#     "datasets",
#     "huggingface_hub",
#     "pandas",
#     "pyarrow",
#     "numpy",
#     "tqdm",
#     "nltk",
#     "langdetect",
#     "immutabledict",
#     "emoji",
#     "syllapy",
#     "absl-py",
#     "ipython",
#     "requests",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="GOLD: Ornith-1.5-9B to Qwen3.5-2B")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GOLD distillation: Ornith-1.5-9B → Qwen3.5-2B

    **Question.** Does on-policy logit distillation (TRL `GOLDTrainer`) from Ornith-1.5-9B improve
    Qwen3.5-2B more than plain fine-tuning on the teacher's outputs?

    **Design.** Two independent experiments, each with its own LoRA adapters:

    | Domain | Train prompts (never the benchmark) | Metric |
    |---|---|---|
    | Instruction following | `allenai/IF_multi_constraints_upto5` (IFEval + IFTrain constraints) | IFBench, prompt-level loose accuracy (58 held-out constraints) |
    | Coding | `deepmind/code_contests` (problems from 2021 and earlier) | LiveCodeBench pass@1, greedy, newest slices |

    **Arms per domain**, all scored on the same items and compared pairwise:
    teacher (reference), student base, student + LoRA SFT on teacher outputs, student + LoRA GOLD.

    Everything runs in non-thinking mode. Each arm is one seed, so treat the result as a first observation.
    Finished stages are cached under `cfg.work_dir`; re-running a cell skips work that is already on disk.

    **Controls** (edit the defaults in the config cell, or set the environment variable before starting):

    - A domain is trained only if the teacher beats the student by `min_gap_points` on the benchmark.
      Set `force_run` (`GOLD_FORCE=1`) to train anyway.
    - `if_gold_minutes` / `code_gold_minutes` cap each GOLD run; `run_if` / `run_code` switch a domain off.
    - Out of memory during training: lower `if_batch_size` or `code_batch_size`.
    - `GOLD_FAST_KERNELS=0` (environment only, then restart) disables the optional linear-attention kernel.
    """)
    return


@app.cell
def _():
    import importlib.util
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    def _site_dirs_with(package: str) -> list[Path]:
        """Directories on the import path that contain `package`, in import order."""
        found = []
        for entry in sys.path:
            folder = Path(entry or ".").resolve()
            if (folder / package / "__init__.py").exists() and folder not in found:
                found.append(folder)
        return found

    def ensure_packages(specs: dict[str, str]) -> None:
        """Install packages whose import name is missing, without ever installing a second copy of torch.

        molab ships a GPU build of torch (with a matching torchvision) outside the notebook's virtual
        environment, where uv cannot see it. Left alone, uv would pull a fresh torch in as a dependency of
        transformers or peft; that copy shadows molab's and no longer matches torchvision. The override file
        tells uv to skip those packages whenever torch is already importable.
        """
        missing = [spec for name, spec in specs.items() if importlib.util.find_spec(name) is None]
        if not missing:
            return
        print("Installing:", missing, flush=True)
        command = ["uv", "pip", "install", "--python", sys.executable]
        if _site_dirs_with("torch"):
            overrides = Path(tempfile.gettempdir()) / "gold_uv_overrides.txt"
            overrides.write_text(
                "".join(f"{name} ; sys_platform == 'never'\n" for name in ("torch", "torchvision", "torchaudio"))
            )
            command += ["--override", str(overrides)]
        try:
            subprocess.run([*command, *missing], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        importlib.invalidate_caches()

    def use_matched_torch() -> None:
        """Repair an environment where torch and torchvision come from different installs.

        Symptom: `RuntimeError: operator torchvision::nms does not exist` while importing transformers or peft.
        """
        torch_dirs, vision_dirs = _site_dirs_with("torch"), _site_dirs_with("torchvision")
        if not torch_dirs or not vision_dirs or torch_dirs[0] == vision_dirs[0]:
            return
        if vision_dirs[0] in torch_dirs:
            # torchvision's own torch is present but shadowed by a newer copy: remove the shadowing copy.
            print(f"Removing the extra torch in {torch_dirs[0]} (torchvision belongs to {vision_dirs[0]}).", flush=True)
            command = ["uv", "pip", "uninstall", "--python", sys.executable, "torch"]
        else:
            # No matching torch anywhere: install a torchvision that matches the active torch.
            print(f"Installing a torchvision that matches the torch in {torch_dirs[0]}.", flush=True)
            command = ["uv", "pip", "install", "--python", sys.executable, "torchvision"]
        subprocess.run(command, check=True)
        importlib.invalidate_caches()
        if "torch" in sys.modules:
            raise RuntimeError(
                "Fixed a torch/torchvision mismatch, but the old torch is still loaded in this session. "
                "Restart the kernel (notebook menu > Restart kernel) and run this cell again."
            )

    ensure_packages(
        {
            "torch": "torch",
            "transformers": "transformers==5.22.2",
            "trl": "trl==1.12.0",
            "peft": "peft",
            "accelerate": "accelerate",
            "datasets": "datasets",
            "huggingface_hub": "huggingface_hub",
            "pandas": "pandas",
            "pyarrow": "pyarrow",
            "nltk": "nltk",
            "langdetect": "langdetect",
            "immutabledict": "immutabledict",
            "emoji": "emoji",
            "syllapy": "syllapy",
            "absl": "absl-py",
            # transformers imports IPython for its Trainer when it detects a notebook runtime such as marimo.
            "IPython": "ipython",
            # trl.experimental.gold imports its vLLM client, which needs requests even when vLLM is unused.
            "requests": "requests",
        }
    )

    # Qwen3.5 runs three of every four layers as linear attention. Without this package transformers uses a
    # slower pure-PyTorch reference path. It must be installed before transformers is imported.
    # Set GOLD_FAST_KERNELS=0 and restart the notebook if the optimized kernel causes errors.
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if os.environ.get("GOLD_FAST_KERNELS", "1") == "1":
        try:
            ensure_packages({"fla": "flash-linear-attention"})
        except subprocess.CalledProcessError as err:
            print(f"Could not install flash-linear-attention ({err}); using the PyTorch fallback.", flush=True)

    use_matched_torch()

    import gc
    import hashlib
    import json
    import math
    import random
    import textwrap
    import time
    from concurrent.futures import ThreadPoolExecutor
    from importlib.metadata import version as _version
    from types import SimpleNamespace

    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        PrinterCallback,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )
    from trl.experimental.gold import GOLDConfig, GOLDTrainer

    print(
        f"torch {torch.__version__} from {Path(torch.__file__).parent.parent} | "
        + " | ".join(f"{name} {_version(name)}" for name in ("transformers", "trl", "peft"))
        + f" | CUDA available: {torch.cuda.is_available()}",
        flush=True,
    )
    if not torch.cuda.is_available():
        print("WARNING: no GPU visible. Attach the GPU from the notebook specs button before running.", flush=True)

    return (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        Dataset,
        GOLDConfig,
        GOLDTrainer,
        LoraConfig,
        Path,
        PeftModel,
        PrinterCallback,
        SimpleNamespace,
        ThreadPoolExecutor,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        ensure_packages,
        gc,
        get_peft_model,
        hashlib,
        importlib,
        json,
        math,
        os,
        pd,
        random,
        set_seed,
        subprocess,
        sys,
        tempfile,
        textwrap,
        time,
        torch,
    )


@app.cell
def _(Path, SimpleNamespace, os):
    def _env(name: str, default):
        """Read an override from the environment, cast to the type of the default."""
        raw = os.environ.get(name)
        if raw is None:
            return default
        if isinstance(default, bool):
            return raw.strip().lower() in ("1", "true", "yes")
        return type(default)(raw)

    cfg = SimpleNamespace(
        teacher_id=_env("GOLD_TEACHER", "ornith-ai/Ornith-1.5-9B"),
        student_id=_env("GOLD_STUDENT", "Qwen/Qwen3.5-2B"),
        work_dir=Path(_env("GOLD_WORK_DIR", "gold_runs")),
        local_data=_env("GOLD_LOCAL_DATA", ""),  # offline mirror of the HF dataset files, for testing
        dtype=_env("GOLD_DTYPE", "bfloat16"),
        seed=_env("GOLD_SEED", 0),
        run_if=_env("GOLD_RUN_IF", True),
        run_code=_env("GOLD_RUN_CODE", True),
        # A domain is trained only if the teacher beats the student by this many points on the benchmark.
        min_gap_points=_env("GOLD_MIN_GAP", 5.0),
        force_run=_env("GOLD_FORCE", False),
        enable_thinking=False,
        gen_batch_size=_env("GOLD_GEN_BS", 48),
        keep_unfinished=_env("GOLD_KEEP_UNFINISHED", False),
        min_train_examples=_env("GOLD_MIN_TRAIN", 64),
        # Instruction following
        if_train_n=_env("GOLD_IF_TRAIN_N", 3000),
        if_max_prompt_tokens=_env("GOLD_IF_PROMPT_TOK", 640),
        if_max_new_tokens=_env("GOLD_IF_NEW_TOK", 512),
        if_eval_max_new_tokens=_env("GOLD_IF_EVAL_TOK", 1024),
        if_gold_minutes=_env("GOLD_IF_MINUTES", 75.0),
        if_batch_size=_env("GOLD_IF_BS", 8),
        if_grad_accum=_env("GOLD_IF_GA", 1),
        # Coding
        code_train_n=_env("GOLD_CODE_TRAIN_N", 2500),
        code_shards=_env("GOLD_CODE_SHARDS", 12),
        code_max_prompt_tokens=_env("GOLD_CODE_PROMPT_TOK", 1024),
        code_max_new_tokens=_env("GOLD_CODE_NEW_TOK", 768),
        code_gold_minutes=_env("GOLD_CODE_MINUTES", 90.0),
        # Code sequences are longer, so the same 8 examples per update are split into two forward passes.
        code_batch_size=_env("GOLD_CODE_BS", 4),
        code_grad_accum=_env("GOLD_CODE_GA", 2),
        code_require_public_tests=_env("GOLD_CODE_REQUIRE_TESTS", True),
        # test5 + test6 are the two newest LiveCodeBench slices (342 problems, 0.7 GB).
        # Use "test.jsonl,test2.jsonl,test3.jsonl,test4.jsonl,test5.jsonl,test6.jsonl" for the full v6 set (4.5 GB).
        lcb_files=_env("GOLD_LCB_FILES", "test5.jsonl,test6.jsonl").split(","),
        lcb_timeout=_env("GOLD_LCB_TIMEOUT", 6),
        # Training (shared by the SFT and GOLD arms)
        lora_r=_env("GOLD_LORA_R", 64),
        lora_alpha=_env("GOLD_LORA_ALPHA", 128),
        lr=_env("GOLD_LR", 1e-4),
        epochs=_env("GOLD_EPOCHS", 1.0),
        logging_steps=_env("GOLD_LOG_STEPS", 10),
        # GOLD only
        lmbda=_env("GOLD_LMBDA", 0.5),  # share of steps where the student writes the text being scored
        beta=_env("GOLD_BETA", 0.5),  # 0 = forward KL, 1 = reverse KL, between = generalized JSD
        temperature=_env("GOLD_TEMPERATURE", 0.9),
        top_p=_env("GOLD_TOP_P", 0.95),
    )
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    return (cfg,)


@app.cell
def _(Path, cfg, json, math, time):
    _t0 = time.time()

    def log(message: str) -> None:
        """Print a message with minutes elapsed since the notebook started."""
        print(f"[{(time.time() - _t0) / 60:6.1f} min] {message}", flush=True)

    def fetch(repo_id: str, filename: str) -> Path:
        """Return a local path to one file of a Hugging Face dataset repo."""
        if cfg.local_data:
            return Path(cfg.local_data) / repo_id.replace("/", "__") / filename
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id, filename, repo_type="dataset"))

    def list_files(repo_id: str, prefix: str) -> list[str]:
        """List files of a Hugging Face dataset repo whose path starts with `prefix`."""
        if cfg.local_data:
            root = Path(cfg.local_data) / repo_id.replace("/", "__")
            names = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]
        else:
            from huggingface_hub import list_repo_files

            names = list_repo_files(repo_id, repo_type="dataset")
        return sorted(name for name in names if name.startswith(prefix))

    def cached_json(path: Path, compute):
        """Load `path` if it exists, otherwise run `compute()` and store its JSON result there."""
        if path.exists():
            return json.loads(path.read_text())
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return value

    def accuracy_ci(flags: list[bool]) -> tuple[float, float]:
        """Accuracy and the half-width of its normal-approximation 95% interval, both in points."""
        n = len(flags)
        p = sum(flags) / n
        return 100 * p, 100 * 1.96 * math.sqrt(p * (1 - p) / n)

    def paired_test(base: list[bool], new: list[bool]) -> dict:
        """Exact McNemar test on per-item results of two models scored on the same items."""
        only_base = sum(1 for a, b in zip(base, new) if a and not b)
        only_new = sum(1 for a, b in zip(base, new) if b and not a)
        n = only_base + only_new
        tail = sum(math.comb(n, i) for i in range(min(only_base, only_new) + 1))
        p_value = 1.0 if n == 0 else min(1.0, 2 * tail / 2**n)
        return {
            "diff_points": 100 * (sum(new) - sum(base)) / len(base),
            "only_base": only_base,
            "only_new": only_new,
            "p_value": p_value,
        }

    return accuracy_ci, cached_json, fetch, list_files, log, paired_test


@app.cell
def _(
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    cfg,
    log,
    torch,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_lm(model_id: str):
        """Load the language model of a Qwen3.5-family checkpoint in `cfg.dtype`."""
        kwargs = {"dtype": getattr(torch, cfg.dtype)}
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
        try:
            model, info = AutoModelForCausalLM.from_pretrained(model_id, output_loading_info=True, **kwargs)
            if info["missing_keys"]:
                raise RuntimeError(f"{len(info['missing_keys'])} missing keys, e.g. {info['missing_keys'][:3]}")
        except Exception as err:  # noqa: BLE001 - any failure falls back to the full multimodal class
            log(f"Text-only load of {model_id} failed ({err}); loading the multimodal class instead.")
            model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        model.eval()
        log(f"Loaded {model_id}: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        return model

    tok = AutoTokenizer.from_pretrained(cfg.student_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher_tok = AutoTokenizer.from_pretrained(cfg.teacher_id)

    teacher = _load_lm(cfg.teacher_id)
    teacher.requires_grad_(False)
    student = _load_lm(cfg.student_id)
    return device, student, teacher, teacher_tok, tok


@app.cell
def _(cfg, mo, student, teacher, teacher_tok, tok):
    def render_prompt(messages: list[dict], tokenizer=None) -> str:
        """Render chat messages into the exact prompt string the student continues from."""
        return (tokenizer or tok).apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=cfg.enable_thinking
        )

    _probe = [{"role": "user", "content": "Reply with the single word: ready"}]
    _student_prompt = render_prompt(_probe)
    _teacher_prompt = render_prompt(_probe, teacher_tok)
    _vocab_student = student.get_input_embeddings().weight.shape[0]
    _vocab_teacher = teacher.get_input_embeddings().weight.shape[0]
    _same_ids = (
        tok(_student_prompt, add_special_tokens=False)["input_ids"]
        == teacher_tok(_student_prompt, add_special_tokens=False)["input_ids"]
    )
    # The standard GOLD loss compares logits token by token, so both models must share one vocabulary.
    if _vocab_student != _vocab_teacher or not _same_ids:
        raise ValueError(
            f"Teacher and student do not share a tokenizer (vocab {_vocab_teacher} vs {_vocab_student}, "
            f"same ids on probe: {_same_ids}). This notebook uses the same-tokenizer GOLD loss."
        )
    template_matches = _student_prompt == _teacher_prompt

    mo.md(
        f"""
    ### Preflight

    - Shared vocabulary: **{_vocab_student:,}** tokens, identical token ids on the probe prompt.
    - Non-thinking prompt identical under both chat templates: **{template_matches}**
      {"" if template_matches else "— the teacher will score text formatted with the student's template; compare the two below."}

    Student prompt: `{_student_prompt!r}`

    Teacher prompt: `{_teacher_prompt!r}`
    """
    )
    return (render_prompt,)


@app.cell
def _(Dataset, cached_json, cfg, hashlib, log, time, tok, torch):
    def generate_texts(model, prompts: list[str], max_new_tokens: int, label: str) -> dict:
        """Greedy batched generation. Returns texts and whether each one ended with an end-of-turn token."""
        model.eval()
        eos = model.generation_config.eos_token_id
        eos_ids = set([eos] if isinstance(eos, int) else (eos or [tok.eos_token_id]))
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        texts, finished = [""] * len(prompts), [False] * len(prompts)
        start = time.time()
        for offset in range(0, len(order), cfg.gen_batch_size):
            batch = order[offset : offset + cfg.gen_batch_size]
            enc = tok([prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.pad_token_id,
                )
            for i, ids in zip(batch, out[:, enc["input_ids"].shape[1] :].tolist()):
                cut = next((k for k, token in enumerate(ids) if token in eos_ids), None)
                finished[i] = cut is not None
                texts[i] = tok.decode(ids if cut is None else ids[:cut], skip_special_tokens=True)
            done = min(offset + cfg.gen_batch_size, len(order))
            log(f"{label}: generated {done}/{len(order)} ({time.time() - start:.0f}s)")
        return {"texts": texts, "finished": finished}

    def cached_generate(tag: str, model, prompts: list[str], max_new_tokens: int) -> dict:
        """`generate_texts` cached on disk, keyed by tag, prompts and token limit."""
        digest = hashlib.sha1("\x00".join(prompts).encode()).hexdigest()[:12]
        path = cfg.work_dir / "generations" / f"{tag}_{max_new_tokens}_{digest}.json"
        return cached_json(path, lambda: generate_texts(model, prompts, max_new_tokens, tag))

    def answer_only(text: str) -> str:
        """Drop a reasoning block if the model produced one."""
        return text.split("</think>")[-1].strip()

    def build_train_dataset(prompts: list[str], completions: list[str], max_prompt_tokens: int, max_new_tokens: int):
        """Tokenize prompt/completion pairs into the pre-tokenized format GOLD's ChatML collator consumes.

        Prompt and completion are tokenized separately and concatenated, which is exactly what on-policy
        generation produces, so the prompt boundary is the same in both the SFT and the GOLD arm.
        """
        rows = []
        for prompt, completion in zip(prompts, completions):
            prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tok(completion, add_special_tokens=False)["input_ids"]
            if not completion_ids or len(prompt_ids) > max_prompt_tokens:
                continue
            completion_ids = completion_ids[: max_new_tokens - 1] + [tok.eos_token_id]
            rows.append(
                {
                    "input_ids": prompt_ids + completion_ids,
                    "completion_mask": [0] * len(prompt_ids) + [1] * len(completion_ids),
                    "original_prompt_text": prompt,
                    "original_completion_text": completion + tok.eos_token,
                }
            )
        return Dataset.from_list(rows) if rows else None

    return answer_only, build_train_dataset, cached_generate


@app.cell
def _(
    GOLDConfig,
    GOLDTrainer,
    LoraConfig,
    PeftModel,
    PrinterCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    cached_json,
    cfg,
    device,
    gc,
    get_peft_model,
    log,
    math,
    set_seed,
    student,
    teacher,
    time,
    tok,
    torch,
):
    class PrintLogs(TrainerCallback):
        """Print the trainer's periodic logs as plain lines."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            shown = {k: v for k, v in (logs or {}).items() if k in ("loss", "on_policy_loss", "off_policy_loss")}
            if shown:
                log(f"step {state.global_step}/{state.max_steps} {shown}")

    class TimeBudget(TrainerCallback):
        """Stop training once a wall-clock budget is used up."""

        def __init__(self, seconds: float):
            self.seconds = seconds
            self.start = None
            self.stopped_at = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.start > self.seconds:
                self.stopped_at = state.global_step
                control.should_training_stop = True

    class MaskedJSDGOLDTrainer(GOLDTrainer):
        """GOLD with the JSD loss computed in float32 on completion positions only.

        The stock loss runs softmax over the full [batch, sequence, vocab] logits in the model dtype. With a
        248k vocabulary in bfloat16 that is both memory-heavy and numerically coarse. Selecting the completion
        positions first and casting them to float32 gives the same value with less memory.
        """

        @staticmethod
        def generalized_jsd_loss(student_logits, teacher_logits, labels=None, **kwargs):
            if labels is not None and not kwargs.get("logits_are_probs", False):
                keep = labels != -100
                student_logits = student_logits[keep].float()
                teacher_logits = teacher_logits[keep].float()
                labels = None
            return GOLDTrainer.generalized_jsd_loss(student_logits, teacher_logits, labels=labels, **kwargs)

    def lora_config() -> LoraConfig:
        return LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.0,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    def make_plan(domain: str) -> dict:
        """Sequence limits, batch shape and GOLD time budget for one domain ("if" or "code")."""
        return {
            "max_prompt_tokens": getattr(cfg, f"{domain}_max_prompt_tokens"),
            "max_new_tokens": getattr(cfg, f"{domain}_max_new_tokens"),
            "batch_size": getattr(cfg, f"{domain}_batch_size"),
            "grad_accum": getattr(cfg, f"{domain}_grad_accum"),
            "minutes": getattr(cfg, f"{domain}_gold_minutes"),
        }

    def _shared_args(tag: str, n_examples: int, plan: dict) -> dict:
        """Optimizer and schedule settings used by both arms, so they differ only in the training signal."""
        steps = math.ceil(n_examples / (plan["batch_size"] * plan["grad_accum"]) * cfg.epochs)
        return {
            "output_dir": str(cfg.work_dir / "trainer" / tag),
            "per_device_train_batch_size": plan["batch_size"],
            "gradient_accumulation_steps": plan["grad_accum"],
            "num_train_epochs": cfg.epochs,
            "learning_rate": cfg.lr,
            "lr_scheduler_type": "cosine",
            "warmup_steps": max(1, int(0.05 * steps)),
            "logging_steps": cfg.logging_steps,
            "save_strategy": "no",
            "report_to": "none",
            "bf16": cfg.dtype == "bfloat16" and device == "cuda",
            "seed": cfg.seed,
            "disable_tqdm": True,
            "dataloader_drop_last": True,
        }

    def train_sft(dataset, tag: str, plan: dict):
        """LoRA fine-tuning with cross-entropy on the teacher's completions (sequence-level distillation)."""
        set_seed(cfg.seed)

        def collate(batch):
            width = max(len(row["input_ids"]) for row in batch)
            input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for i, row in enumerate(batch):
                n = len(row["input_ids"])
                ids = torch.tensor(row["input_ids"], dtype=torch.long)
                input_ids[i, :n] = ids
                attention_mask[i, :n] = 1
                labels[i, :n] = torch.where(torch.tensor(row["completion_mask"], dtype=torch.bool), ids, -100)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        model = get_peft_model(student, lora_config())
        args = TrainingArguments(remove_unused_columns=False, **_shared_args(tag, len(dataset), plan))
        trainer = Trainer(
            model=model, args=args, train_dataset=dataset, data_collator=collate, callbacks=[PrintLogs()]
        )
        trainer.remove_callback(PrinterCallback)
        trainer.train()
        return model, {"steps": trainer.state.global_step, "stopped_early": False}

    def train_gold(dataset, tag: str, plan: dict):
        """LoRA training with GOLD: the teacher's token distribution is matched on teacher- and student-written text."""
        set_seed(cfg.seed)
        budget = TimeBudget(plan["minutes"] * 60)
        args = GOLDConfig(
            max_length=plan["max_prompt_tokens"] + plan["max_new_tokens"],
            max_completion_length=plan["max_new_tokens"],
            lmbda=plan.get("lmbda", cfg.lmbda),
            beta=cfg.beta,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            use_uld_loss=False,
            gradient_checkpointing=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            **_shared_args(tag, len(dataset), plan),
        )
        trainer = MaskedJSDGOLDTrainer(
            model=student,
            teacher_model=teacher,
            args=args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=lora_config(),
            callbacks=[PrintLogs(), budget],
        )
        trainer.remove_callback(PrinterCallback)
        trainer.train()
        info = {"steps": trainer.state.global_step, "stopped_early": budget.stopped_at is not None}
        return trainer.model, info

    def release(model) -> None:
        """Remove LoRA layers from the shared student so the next arm starts from the base weights."""
        if isinstance(model, PeftModel):
            model.unload()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    def run_arm(domain: str, arm: str, train_fn, eval_fn) -> dict:
        """Train one arm (or reload its saved adapter), evaluate it, and cache the result."""
        tag = f"{domain}_{arm}"
        adapter_dir = cfg.work_dir / "adapters" / tag

        def compute():
            info = {}
            if (adapter_dir / "adapter_config.json").exists():
                model = PeftModel.from_pretrained(student, str(adapter_dir))
            else:
                start = time.time()
                model, info = train_fn(tag)
                info["train_minutes"] = round((time.time() - start) / 60, 1)
                model.save_pretrained(str(adapter_dir))
                log(f"{tag}: trained {info}")
            try:
                result = eval_fn(model, tag)
            finally:
                release(model)
            return {**result, "train": info}

        return cached_json(cfg.work_dir / "results" / f"{tag}.json", compute)

    return make_plan, release, run_arm, train_gold, train_sft


@app.cell
def _(
    answer_only,
    build_train_dataset,
    cached_generate,
    device,
    log,
    mo,
    release,
    render_prompt,
    teacher,
    time,
    torch,
    train_gold,
):
    # Smoke test: push a handful of examples through the whole GOLD path before the long stages start.
    _tasks = [
        "Write two sentences about the sea.",
        "List three uses of a brick.",
        "Explain what a prime number is in one sentence.",
        "Write a haiku about rain.",
        "Give one tip for learning to cook.",
        "Describe the colour blue to a child.",
        "Name two planets and one fact about each.",
        "Write a short thank-you note to a teacher.",
    ] * 2
    _prompts = [render_prompt([{"role": "user", "content": task}]) for task in _tasks]
    _generated = cached_generate("smoke_teacher", teacher, _prompts, 64)
    _completions = [answer_only(text) for text in _generated["texts"]]
    _dataset = build_train_dataset(_prompts, _completions, 128, 96)
    if _dataset is None or len(_dataset) < 8:
        raise RuntimeError("Smoke test could not build 8 training examples from the teacher's outputs.")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _start = time.time()
    # lmbda=1 makes both steps on-policy, so student generation and teacher scoring are both exercised.
    _plan = {"max_prompt_tokens": 128, "max_new_tokens": 96, "batch_size": 4, "grad_accum": 1, "minutes": 10, "lmbda": 1.0}
    _model, _info = train_gold(_dataset.select(range(8)), "smoke_gold", _plan)
    release(_model)
    smoke = {
        "steps": _info["steps"],
        "seconds_per_step": round((time.time() - _start) / max(1, _info["steps"]), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if device == "cuda" else None,
        "teacher_sample": _completions[0][:200],
    }
    log(f"Smoke test passed: {smoke}")
    mo.md(
        f"""
    ### Smoke test passed

    GOLD ran **{smoke['steps']} steps** on 8 tiny examples
    ({smoke['seconds_per_step']} s/step including start-up, peak VRAM {smoke['peak_vram_gb']} GB).
    Real batches are larger, so expect a higher peak in the full runs.

    Teacher sample: `{smoke['teacher_sample']!r}`
    """
    )
    return (smoke,)


@app.cell
def _(
    Path,
    answer_only,
    cached_generate,
    cfg,
    ensure_packages,
    importlib,
    json,
    render_prompt,
    subprocess,
    sys,
):
    def _load_ifbench():
        """Import the IFBench verifiers (PyPI package, or a clone of the repository as a fallback)."""
        if importlib.util.find_spec("ifbench") is None:
            try:
                ensure_packages({"ifbench": "ifbench"})
            except subprocess.CalledProcessError:
                repo = cfg.work_dir / "IFBench"
                if not repo.exists():
                    subprocess.run(
                        ["git", "clone", "--depth", "1", "https://github.com/allenai/IFBench.git", str(repo)],
                        check=True,
                    )
                sys.path.insert(0, str(repo))
        import ifbench
        from ifbench import instructions_registry

        with open(Path(ifbench.data_path())) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return rows, instructions_registry.INSTRUCTION_DICT

    def _loose_variants(response: str) -> list[str]:
        """The response variants IFEval/IFBench accept for the loose metric."""
        lines = response.split("\n")
        variants = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
        return variants + [variant.replace("*", "") for variant in variants]

    def _score_ifbench_row(row: dict, response: str, registry: dict) -> dict:
        """Strict and loose verdicts for one prompt, following IFBench's `evaluation_lib`."""
        strict, loose = [], []
        for instruction_id, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            checker = registry[instruction_id](instruction_id)
            checker.build_description(**{k: v for k, v in kwargs.items() if v is not None})
            arg_names = checker.get_instruction_args()
            if arg_names and "prompt" in arg_names:
                checker.build_description(prompt=row["prompt"])

            def follows(text: str) -> bool:
                try:
                    return bool(text.strip()) and bool(checker.check_following(text))
                except Exception:  # noqa: BLE001 - a verifier crash on odd text counts as not followed
                    return False

            strict.append(follows(response))
            loose.append(any(follows(variant) for variant in _loose_variants(response)))
        return {"strict": all(strict), "loose": all(loose)}

    if cfg.run_if:
        ifb_rows, _registry = _load_ifbench()
    else:
        ifb_rows, _registry = [], {}

    def eval_ifbench(model, tag: str) -> dict:
        """Generate answers to all IFBench prompts and score them. `flags` holds the loose verdict per prompt."""
        prompts = [render_prompt([{"role": "user", "content": row["prompt"]}]) for row in ifb_rows]
        generated = cached_generate(f"ifbench_{tag}", model, prompts, cfg.if_eval_max_new_tokens)
        verdicts = [
            _score_ifbench_row(row, answer_only(text), _registry) for row, text in zip(ifb_rows, generated["texts"])
        ]
        return {
            "metric": "IFBench prompt-level loose accuracy",
            "flags": [v["loose"] for v in verdicts],
            "strict_flags": [v["strict"] for v in verdicts],
        }

    return eval_ifbench, ifb_rows


@app.cell
def _(accuracy_ci, cached_json, cfg, log, mo, paired_test, pd):
    def reference_eval(domain: str, name: str, model, eval_fn) -> dict:
        """Evaluate an untrained model (teacher or student base) once and cache the result."""
        tag = f"{domain}_{name}"
        return cached_json(cfg.work_dir / "results" / f"{tag}.json", lambda: eval_fn(model, tag))

    def gap_decision(teacher_result: dict, student_result: dict) -> dict:
        """Compare teacher and student on the benchmark and decide whether the domain is worth training."""
        teacher_acc, _ = accuracy_ci(teacher_result["flags"])
        student_acc, _ = accuracy_ci(student_result["flags"])
        gap = teacher_acc - student_acc
        return {
            "teacher": round(teacher_acc, 1),
            "student": round(student_acc, 1),
            "gap": round(gap, 1),
            "go": cfg.force_run or gap >= cfg.min_gap_points,
        }

    def results_view(title: str, results: dict):
        """Accuracy table plus paired comparisons for whatever arms finished."""
        rows = []
        for name, result in results.items():
            if result is None:
                continue
            acc, half = accuracy_ci(result["flags"])
            train = result.get("train") or {}
            rows.append(
                {
                    "model": name,
                    "accuracy": round(acc, 1),
                    "ci95_half_width": round(half, 1),
                    "items": len(result["flags"]),
                    "train_steps": train.get("steps"),
                    "train_minutes": train.get("train_minutes"),
                    "stopped_by_time_budget": train.get("stopped_early"),
                }
            )
        comparisons = []
        for base, new in (("student base", "student + SFT"), ("student base", "student + GOLD"), ("student + SFT", "student + GOLD")):
            if results.get(base) and results.get(new):
                test = paired_test(results[base]["flags"], results[new]["flags"])
                comparisons.append(
                    {
                        "comparison": f"{new} vs {base}",
                        "diff_points": round(test["diff_points"], 1),
                        "only_first_correct": test["only_new"],
                        "only_second_correct": test["only_base"],
                        "mcnemar_p": round(test["p_value"], 4),
                    }
                )
        # Also print the tables, so they are visible in the logs and in the single-cell version.
        log(f"{title}\n{pd.DataFrame(rows).to_string(index=False)}")
        if comparisons:
            log(f"Paired comparisons\n{pd.DataFrame(comparisons).to_string(index=False)}")
        parts = [mo.md(f"### {title}"), mo.ui.table(rows, selection=None)]
        if comparisons:
            parts.append(mo.ui.table(comparisons, selection=None))
        return mo.vstack(parts), {"accuracy": rows, "comparisons": comparisons}

    return gap_decision, reference_eval, results_view


@app.cell
def _(
    cfg,
    eval_ifbench,
    gap_decision,
    log,
    mo,
    reference_eval,
    smoke,
    student,
    teacher,
):
    # Instruction following, step 1: is the teacher actually better than the student on IFBench?
    if cfg.run_if and smoke:
        if_student = reference_eval("if", "student_base", student, eval_ifbench)
        if_teacher = reference_eval("if", "teacher", teacher, eval_ifbench)
        if_gate = gap_decision(if_teacher, if_student)
        log(f"IFBench go/no-go: {if_gate}")
        _view = mo.md(
            f"""
    ### IFBench go/no-go

    Teacher **{if_gate['teacher']}** vs student **{if_gate['student']}** (gap {if_gate['gap']} points,
    threshold {cfg.min_gap_points}). Training this domain: **{if_gate['go']}**
    """
        )
    else:
        if_student, if_teacher, if_gate = None, None, {"go": False}
        _view = mo.md("### IFBench skipped (`cfg.run_if` is off)")
    _view
    return if_gate, if_student, if_teacher


@app.cell
def _(
    answer_only,
    build_train_dataset,
    cached_generate,
    cfg,
    fetch,
    if_gate,
    ifb_rows,
    log,
    mo,
    pd,
    random,
    render_prompt,
    teacher,
):
    def _if_train_prompts() -> list[str]:
        """Sample training prompts built from IFEval + IFTrain constraints, none of them an IFBench prompt."""
        frame = pd.read_parquet(
            fetch("allenai/IF_multi_constraints_upto5", "data/train-00000-of-00001.parquet"), columns=["messages"]
        )
        benchmark = {row["prompt"].strip() for row in ifb_rows}
        users = []
        for messages in frame["messages"]:
            content = messages[0]["content"]
            if 40 <= len(content) <= 2000 and content.strip() not in benchmark:
                users.append(content)
        users = sorted(set(users))
        random.Random(cfg.seed).shuffle(users)
        return users[: cfg.if_train_n]

    if if_gate["go"]:
        _users = _if_train_prompts()
        _prompts = [render_prompt([{"role": "user", "content": user}]) for user in _users]
        _generated = cached_generate("if_train_teacher", teacher, _prompts, cfg.if_max_new_tokens)
        _pairs = [
            (prompt, answer_only(text))
            for prompt, text, done in zip(_prompts, _generated["texts"], _generated["finished"])
            if done or cfg.keep_unfinished
        ]
        if_train_ds = build_train_dataset(
            [p for p, _ in _pairs], [c for _, c in _pairs], cfg.if_max_prompt_tokens, cfg.if_max_new_tokens
        )
        _n = 0 if if_train_ds is None else len(if_train_ds)
        log(f"IF train set: {_n} examples from {len(_users)} prompts")
        if _n < cfg.min_train_examples:
            if_train_ds = None
        _view = mo.md(f"### IF training set: {_n} examples (from {len(_users)} sampled prompts)")
    else:
        if_train_ds = None
        _view = mo.md("### IF training set skipped")
    _view
    return (if_train_ds,)


@app.cell
def _(eval_ifbench, if_train_ds, make_plan, run_arm, train_gold, train_sft):
    # Both arms see the same examples, batch shape, learning rate and schedule; only the loss differs.
    if if_train_ds is not None:
        _plan = make_plan("if")
        if_sft = run_arm("if", "sft", lambda tag: train_sft(if_train_ds, tag, _plan), eval_ifbench)
        if_gold = run_arm("if", "gold", lambda tag: train_gold(if_train_ds, tag, _plan), eval_ifbench)
    else:
        if_sft, if_gold = None, None
    return if_gold, if_sft


@app.cell
def _(if_gold, if_sft, if_student, if_teacher, mo, results_view):
    if if_student is not None:
        _view, if_summary = results_view(
            "Instruction following: IFBench, prompt-level loose accuracy (%)",
            {"teacher": if_teacher, "student base": if_student, "student + SFT": if_sft, "student + GOLD": if_gold},
        )
    else:
        _view, if_summary = mo.md("### Instruction following: not run"), None
    _view
    return (if_summary,)


@app.cell
def _(
    answer_only,
    cached_generate,
    cached_json,
    cfg,
    fetch,
    if_summary,
    importlib,
    json,
    os,
    render_prompt,
    subprocess,
    sys,
    textwrap,
):
    # The coding experiment starts only after the instruction-following cells above have finished.
    code_ready = cfg.run_code and (if_summary is not None or not cfg.run_if)

    _LCB_SYSTEM = (
        "You are an expert Python programmer. You will be given a question (problem specification) and will "
        "generate a correct Python program that matches the specification and passes all tests."
    )

    def code_messages(question: str, starter_code: str = "") -> list[dict]:
        """LiveCodeBench's generic code-generation prompt, used for both training and evaluation."""
        user = f"### Question:\n{question}\n\n"
        if starter_code:
            user += (
                "### Format: You will use the following starter code to write the solution to the problem and "
                f"enclose your code within delimiters.\n```python\n{starter_code}\n```\n\n"
            )
        else:
            user += (
                "### Format: Read the inputs from stdin solve the problem and write the answer to stdout (do not "
                "directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that "
                "when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
                "```python\n# YOUR CODE HERE\n```\n\n"
            )
        user += "### Answer: (use the provided format with backticks)\n\n"
        return [{"role": "system", "content": _LCB_SYSTEM}, {"role": "user", "content": user}]

    def extract_code(text: str) -> str:
        """The last fenced block of an answer, as LiveCodeBench extracts it."""
        lines = text.split("\n")
        fences = [i for i, line in enumerate(lines) if "```" in line]
        return "" if len(fences) < 2 else "\n".join(lines[fences[-2] + 1 : fences[-1]])

    _WORKER = '''
    import dataclasses
    import json
    import sys

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics


    def main():
        codes_path, out_path, num_proc, timeout, *problem_paths = sys.argv[1:]
        with open(codes_path) as handle:
            codes = json.load(handle)
        names = {field.name for field in dataclasses.fields(CodeGenerationProblem)}
        problems = []
        for path in problem_paths:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["question_id"] in codes:
                        problems.append(CodeGenerationProblem(**{k: v for k, v in row.items() if k in names}))
        samples = [problem.get_evaluation_sample() for problem in problems]
        generations = [[codes[problem.question_id]] for problem in problems]
        _, results, _ = codegen_metrics(
            samples, generations, k_list=[1], num_process_evaluate=int(num_proc), timeout=int(timeout)
        )
        graded = {
            problem.question_id: bool(all(outcome > 0 for outcome in results[index][0]))
            for index, problem in enumerate(problems)
        }
        with open(out_path, "w") as handle:
            json.dump(graded, handle)


    if __name__ == "__main__":
        main()
    '''

    def _load_lcb():
        """Read the selected LiveCodeBench files and make its evaluator importable."""
        paths = [fetch("livecodebench/code_generation_lite", name) for name in cfg.lcb_files]
        problems = []
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    problems.append({k: row[k] for k in ("question_id", "question_content", "starter_code", "difficulty")})
        repo = None
        if importlib.util.find_spec("lcb_runner") is None:
            repo = cfg.work_dir / "LiveCodeBench"
            if not repo.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/LiveCodeBench/LiveCodeBench.git", str(repo)],
                    check=True,
                )
        worker = cfg.work_dir / "lcb_worker.py"
        worker.write_text(textwrap.dedent(_WORKER))
        return problems, paths, repo, worker

    if code_ready:
        _lcb_problems, _lcb_paths, _lcb_repo, _lcb_worker = _load_lcb()
    else:
        _lcb_problems, _lcb_paths, _lcb_repo, _lcb_worker = [], [], None, None

    def _grade(tag: str, codes: dict) -> dict:
        """Run LiveCodeBench's own test executor in a separate process. Returns pass/fail per question id."""
        codes_path = cfg.work_dir / "lcb" / f"{tag}_codes.json"
        codes_path.parent.mkdir(parents=True, exist_ok=True)
        codes_path.write_text(json.dumps(codes))
        graded_path = cfg.work_dir / "lcb" / f"{tag}_graded.json"
        env = dict(os.environ)
        if _lcb_repo is not None:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_lcb_repo), env.get("PYTHONPATH", "")]))
        command = [sys.executable, str(_lcb_worker), str(codes_path), str(graded_path)]
        command += [str(os.cpu_count() or 4), str(cfg.lcb_timeout), *map(str, _lcb_paths)]
        subprocess.run(command, check=True, env=env)
        return json.loads(graded_path.read_text())

    def eval_lcb(model, tag: str) -> dict:
        """Greedy pass@1 on the selected LiveCodeBench problems."""
        prompts = [render_prompt(code_messages(p["question_content"], p["starter_code"])) for p in _lcb_problems]
        generated = cached_generate(f"lcb_{tag}", model, prompts, cfg.code_max_new_tokens)
        codes = {p["question_id"]: extract_code(answer_only(t)) for p, t in zip(_lcb_problems, generated["texts"])}
        graded = cached_json(cfg.work_dir / "lcb" / f"{tag}_verdicts.json", lambda: _grade(tag, codes))
        return {
            "metric": "LiveCodeBench pass@1 (greedy)",
            "flags": [bool(graded[p["question_id"]]) for p in _lcb_problems],
            "difficulty": [p["difficulty"] for p in _lcb_problems],
        }

    return code_messages, code_ready, eval_lcb, extract_code


@app.cell
def _(
    cfg,
    code_ready,
    eval_lcb,
    gap_decision,
    log,
    mo,
    reference_eval,
    student,
    teacher,
):
    # Coding, step 1: is the teacher actually better than the student on LiveCodeBench?
    if code_ready:
        code_student = reference_eval("code", "student_base", student, eval_lcb)
        code_teacher = reference_eval("code", "teacher", teacher, eval_lcb)
        code_gate = gap_decision(code_teacher, code_student)
        log(f"LiveCodeBench go/no-go: {code_gate}")
        _view = mo.md(
            f"""
    ### LiveCodeBench go/no-go

    Teacher **{code_gate['teacher']}** vs student **{code_gate['student']}** (gap {code_gate['gap']} points,
    threshold {cfg.min_gap_points}). Training this domain: **{code_gate['go']}**
    """
        )
    else:
        code_student, code_teacher, code_gate = None, None, {"go": False}
        _view = mo.md("### LiveCodeBench skipped")
    _view
    return code_gate, code_student, code_teacher


@app.cell
def _(
    Path,
    ThreadPoolExecutor,
    answer_only,
    build_train_dataset,
    cached_generate,
    cached_json,
    cfg,
    code_gate,
    code_messages,
    extract_code,
    fetch,
    list_files,
    log,
    mo,
    os,
    random,
    render_prompt,
    subprocess,
    sys,
    teacher,
    tempfile,
):
    def _contest_problems() -> list[dict]:
        """Competitive-programming problems from CodeContests (all published before LiveCodeBench's window)."""
        import pyarrow.parquet as pq

        columns = ["name", "description", "public_tests", "cf_rating", "is_description_translated"]
        problems = []
        for shard in list_files("deepmind/code_contests", "data/train-")[: cfg.code_shards]:
            for row in pq.read_table(fetch("deepmind/code_contests", shard), columns=columns).to_pylist():
                tests = row["public_tests"] or {}
                if row["is_description_translated"] or not tests.get("input"):
                    continue
                if not 200 <= len(row["description"]) <= 3500 or (row["cf_rating"] or 0) > 1900:
                    continue
                problems.append(
                    {"name": row["name"], "description": row["description"], "inputs": tests["input"], "outputs": tests["output"]}
                )
            log(f"CodeContests: {len(problems)} usable problems after {shard}")
        problems.sort(key=lambda problem: problem["name"])
        random.Random(cfg.seed).shuffle(problems)
        return problems[: cfg.code_train_n]

    def _passes_public_tests(code: str, inputs: list[str], outputs: list[str]) -> bool:
        """Run a stdin/stdout solution on the problem's public examples."""
        if not code.strip():
            return False
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "solution.py"
            script.write_text(code)
            for given, expected in zip(inputs, outputs):
                try:
                    run = subprocess.run(
                        [sys.executable, str(script)], input=given, capture_output=True, text=True, timeout=5, cwd=folder
                    )
                except subprocess.TimeoutExpired:
                    return False
                if run.returncode != 0 or run.stdout.split() != expected.split():
                    return False
        return True

    if code_gate["go"]:
        _problems = _contest_problems()
        _prompts = [render_prompt(code_messages(problem["description"])) for problem in _problems]
        _generated = cached_generate("code_train_teacher", teacher, _prompts, cfg.code_max_new_tokens)
        _answers = [answer_only(text) for text in _generated["texts"]]

        def _verify() -> list[bool]:
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
                return list(
                    pool.map(
                        lambda pair: _passes_public_tests(extract_code(pair[1]), pair[0]["inputs"], pair[0]["outputs"]),
                        zip(_problems, _answers),
                    )
                )

        _passed = cached_json(cfg.work_dir / "code_train_public_tests.json", _verify)
        _keep = [
            (done or cfg.keep_unfinished) and (ok or not cfg.code_require_public_tests)
            for done, ok in zip(_generated["finished"], _passed)
        ]
        code_train_ds = build_train_dataset(
            [p for p, k in zip(_prompts, _keep) if k],
            [a for a, k in zip(_answers, _keep) if k],
            cfg.code_max_prompt_tokens,
            cfg.code_max_new_tokens,
        )
        _n = 0 if code_train_ds is None else len(code_train_ds)
        _rate = 100 * sum(_passed) / max(1, len(_passed))
        log(f"Code train set: {_n} examples; teacher passed public tests on {_rate:.0f}% of {len(_problems)} problems")
        if _n < cfg.min_train_examples:
            code_train_ds = None
        _view = mo.md(
            f"### Code training set: {_n} examples "
            f"(teacher solved the public tests of {_rate:.0f}% of {len(_problems)} problems)"
        )
    else:
        code_train_ds = None
        _view = mo.md("### Code training set skipped")
    _view
    return (code_train_ds,)


@app.cell
def _(code_train_ds, eval_lcb, make_plan, run_arm, train_gold, train_sft):
    if code_train_ds is not None:
        _plan = make_plan("code")
        code_sft = run_arm("code", "sft", lambda tag: train_sft(code_train_ds, tag, _plan), eval_lcb)
        code_gold = run_arm("code", "gold", lambda tag: train_gold(code_train_ds, tag, _plan), eval_lcb)
    else:
        code_sft, code_gold = None, None
    return code_gold, code_sft


@app.cell
def _(code_gold, code_sft, code_student, code_teacher, mo, results_view):
    if code_student is not None:
        _view, code_summary = results_view(
            "Coding: LiveCodeBench pass@1, greedy (%)",
            {"teacher": code_teacher, "student base": code_student, "student + SFT": code_sft, "student + GOLD": code_gold},
        )
    else:
        _view, code_summary = mo.md("### Coding: not run"), None
    _view
    return (code_summary,)


@app.cell
def _(cfg, code_summary, if_summary, json, log, mo, os, smoke):
    # Final record of the run. Set HF_TOKEN and GOLD_HUB_REPO to also upload results and adapters to the Hub.
    _summary = {
        "teacher": cfg.teacher_id,
        "student": cfg.student_id,
        "config": {k: (str(v) if not isinstance(v, (int, float, bool, str, list)) else v) for k, v in vars(cfg).items()},
        "smoke_test": smoke,
        "instruction_following": if_summary,
        "coding": code_summary,
    }
    _path = cfg.work_dir / "summary.json"
    _path.write_text(json.dumps(_summary, indent=2))
    log(f"Wrote {_path}")
    _uploaded = ""
    if os.environ.get("HF_TOKEN") and os.environ.get("GOLD_HUB_REPO"):
        from huggingface_hub import HfApi

        _api = HfApi(token=os.environ["HF_TOKEN"])
        _api.create_repo(os.environ["GOLD_HUB_REPO"], private=True, exist_ok=True)
        _api.upload_folder(
            repo_id=os.environ["GOLD_HUB_REPO"],
            folder_path=str(cfg.work_dir),
            allow_patterns=["summary.json", "results/*", "adapters/**"],
        )
        _uploaded = f" and uploaded to `{os.environ['GOLD_HUB_REPO']}`"
    mo.md(
        f"""
    ### Done

    Summary written to `{_path}`{_uploaded}. Each arm is a single seed: differences smaller than the
    confidence intervals above are not evidence of an effect.
    """
    )
    return


if __name__ == "__main__":
    app.run()
