import json, re, sys, collections
for name in ("bf16", "fp8_scaled", "int8_convrot"):
    h = json.load(open(f"krea2_turbo_{name}.header.json", encoding="utf-8"))
    meta = h.pop("__metadata__", None)
    print("=" * 70); print(name, "tensors:", len(h), "metadata keys:", list(meta)[:8] if meta else None)
    dt = collections.Counter(v["dtype"] for v in h.values()); print(" dtypes:", dict(dt))
    suffixes = collections.Counter(k.rsplit(".", 1)[-1] for k in h); print(" suffixes:", dict(suffixes.most_common(12)))
    pats = collections.Counter(re.sub(r"\.\d+\.", ".N.", k) for k in h)
    print(" key patterns:", len(pats))
    for p, c in sorted(pats.items()):
        v = h[next(k for k in h if re.sub(r"\.\d+\.", ".N.", k) == p)]
        print(f"   x{c:<3d} {v['dtype']:8s} {str(v['shape']):22s} {p}")
    marker = [k for k in h if "scaled_fp8" in k or k.endswith("comfy_quant")]
    if marker:
        print(" quant markers:", len(marker), "e.g.", marker[:2])
    if meta: print(" metadata sample:", {k: str(v)[:80] for k, v in list(meta.items())[:5]})
