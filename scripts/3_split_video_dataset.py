import os
import re
import shutil
import random
from collections import defaultdict

# -- Config --
source_root = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/Sample_V2"
dest_root = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL_V2"

# Choose one of: "percentage", "signer", "combined"
split_mode = "percentage"

# Percentage split ratios (only used in "percentage" or "combined")
split_percentages = {"train": 0.8, "dev": 0.1, "test": 0.1}

# Signer-based assignment override (only used in "signer" or "combined")
# Every signer you want to assign must be listed here when in "signer" mode.
signer_split_map = {
    "Signer3": "test"
}

# Whether to shuffle videos before splitting (applies to percentage splits)
use_random_order = True

# Records & counters
copy_counts = defaultdict(int)
sentence_counts = defaultdict(lambda: defaultdict(int))
split_record = defaultdict(lambda: defaultdict(list))


def numeric_key(filename):
    nums = re.findall(r"(\d+)", filename)
    return int(nums[-1]) if nums else 0


def compute_counts(N, percentages):
    """
    Given total N and a dict of two percentages (e.g. {'train':0.8,'dev':0.2}),
    returns integer counts so that:
      - dev_count = floor(N*p_dev) + (1 if frac >= 0.5 else 0)
      - train_count = N - dev_count
    """
    # keep only splits with p>0
    p = {k: v for k, v in percentages.items() if v > 0}
    if set(p.keys()) == {"train", "dev"}:
        p_train, p_dev = p["train"], p["dev"]
        exact_dev = N * p_dev
        floor_dev = int(exact_dev)
        frac = exact_dev - floor_dev
        # your rule: fractional part <0.5 → floor, ≥0.5 → ceil
        dev_count = floor_dev + (1 if frac >= 0.5 else 0)
        train_count = N - dev_count
        return {"train": train_count, "dev": dev_count}
    else:
        # fallback: round each then adjust
        raw = {k: int(round(N * v)) for k, v in p.items()}
        total = sum(raw.values())
        # same post‐adjustment as before
        if total > N:
            for k in sorted(raw, key=lambda k: raw[k], reverse=True):
                if raw[k] > 0 and total > N:
                    raw[k] -= 1
                    total -= 1
        elif total < N:
            for k in raw:
                if total < N:
                    raw[k] += 1
                    total += 1
        return raw


print(f"Running split_mode = {split_mode!r}\n")

for sentence in sorted(os.listdir(source_root)):
    s_path = os.path.join(source_root, sentence)
    if not os.path.isdir(s_path):
        continue

    print(f"Sentence: {sentence}")
    for signer in sorted(os.listdir(s_path)):
        signer_path = os.path.join(s_path, signer)
        if not os.path.isdir(signer_path):
            continue

        vids = [f for f in os.listdir(signer_path)
                if os.path.isfile(os.path.join(signer_path, f)) and not f.startswith('.')]
        if not vids:
            print(f"  [WARNING] {signer} empty, skipping.")
            continue

        # Order or shuffle
        if use_random_order:
            random.shuffle(vids)
        else:
            vids.sort(key=numeric_key)

        assignment = defaultdict(list)

        if split_mode == "combined" or split_mode == "signer":
            # Signer override logic
            if signer in signer_split_map:
                target = signer_split_map[signer]
                assignment[target] = vids[:]
                print(f"  [OVERRIDE] {signer} → {target.upper()} ({len(vids)} videos)")
            elif split_mode == "signer":
                # In pure signer mode, every signer must be mapped
                print(f"  [ERROR] signer {signer} not in signer_split_map; skipping.")
                continue

        if split_mode == "combined" or split_mode == "percentage":
            # Percentage split logic (only for unmapped signers in combined,
            # or for all signers in pure percentage mode)
            to_split = vids[:]
            if split_mode == "combined" and signer in signer_split_map:
                to_split = []  # already handled by override

            N = len(to_split)
            if N > 0:
                if N < 2:
                    print(f"  [WARNING] {signer} has {N} videos—cannot split by percentage.")
                    # In percentage-only mode, you might want to send them all to train:
                    if split_mode == "percentage":
                        assignment["train"] = to_split
                else:
                    counts = compute_counts(N, split_percentages)
                    idx = 0
                    for split_name, cnt in counts.items():
                        for _ in range(cnt):
                            if idx < N:
                                assignment[split_name].append(to_split[idx])
                                idx += 1
                    # Any remainder goes to train
                    while idx < N:
                        assignment["train"].append(to_split[idx])
                        idx += 1
                    print(f"  [PERCENT] {signer} → "
                          f"train={len(assignment['train'])}, "
                          f"dev={len(assignment['dev'])}")

        # Copy and record
        for split_name, files in assignment.items():
            for f in files:
                src = os.path.join(signer_path, f)
                dst_dir = os.path.join(dest_root, split_name, sentence)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(dst_dir, f))

                copy_counts[split_name] += 1
                sentence_counts[split_name][sentence] += 1
                split_record[split_name][sentence].append(f)
                print(f"    [{split_name.upper():5s}] {f}")
    print()

# Save record
out_txt = os.path.join(dest_root, f"split_record_{split_mode}.txt")
with open(out_txt, "w", encoding="utf-8") as fp:
    for split in ("train", "dev", "test"):
        fp.write(f"{split.upper()}:\n")
        for sent, lst in split_record[split].items():
            fp.write(f"  {sent}:\n")
            for fn in lst:
                fp.write(f"    - {fn}\n")
        fp.write("\n")

# Summary
print("=== Summary ===")
for split in ("train", "dev", "test"):
    print(f"{split.upper():5s} total: {copy_counts[split]} videos")
    for sent, cnt in sentence_counts[split].items():
        print(f"   {sent:20s}: {cnt}")
print(f"\nSplit record saved to: {out_txt}")
