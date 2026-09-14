# Krea 2 Merge Tool

Merges, analyzes, extracts and converts Krea 2 diffusion models on Windows, with a GUI and a CLI.

- **LoRA merge**: several LoRA or LoKr files into one LoRA, exact when the inputs are ordinary LoRAs, with per LoRA strength and block shaping.
- **Analysis**: the rank needed and the energy retained, per layer group, with a quantization noise floor.
- **Extract**: a LoRA from the difference between two checkpoints in any storage format.
- **Checkpoint merge and convert**: one to three checkpoints and up to four LoRAs, nine merge methods, output in fp16, bf16, fp32, fp8, fp8 scaled or int8 convrot in the exact layout of the official Krea 2 files, or the result written as a LoRA.

Every run is described by a recipe. The recipe is stored in the output file's metadata, can be reloaded into the GUI, and reproduces the file byte for byte.

![Checkpoint merge tab, light theme](scr/light.png)

![LoRA merge tab with block shaping and an analysis in the log, dark theme](scr/dark.png)

## Install and run

1. Install Python 3.12 from python.org with "Add Python to PATH" and tcl/tk enabled.
2. Run `install.bat`. It creates `venv`, installs torch from the PyTorch CUDA index (never the CPU only PyPI build), installs the other dependencies and runs a self check.
3. Run `run.bat` for the GUI. `run.bat --help` shows the CLI.

The GUI follows the display DPI and the Windows text size setting (Settings > Accessibility > Text size), which classic Windows programs otherwise ignore. The Scale box in the top right overrides it with a fixed 100 to 200 percent; the choice takes effect at the next start. Theme and scale are remembered in `settings.json` next to the tool, or given on the command line with `--theme` and `--scale`.

The tool needs a Krea 2 diffusion model file. The text encoder and the VAE are separate files and are never touched.

## Files it reads and writes

| Format | Read | Write |
|---|---|---|
| bf16, fp16, fp32 | yes | yes |
| fp8 plain | yes | yes, with a warning: small weights are lost |
| fp8 scaled (official Krea 2 layout, legacy ComfyUI layout, per tensor descriptors) | yes | official layout |
| int8 convrot (official Krea 2 layout) | yes | yes |
| int8 without rotation | yes | no |
| LoRA: ComfyUI, kohya, musubi-tuner, diffusers, PEFT | yes | ComfyUI or kohya naming |
| LoKr (ai-toolkit, full or decomposed factors) | yes | converted to a LoRA by SVD |
| DoRA | refused | no |

Quantized outputs reproduce the official files: converting the official bf16 file gives the official fp8 scaled file bit for bit, and the official int8 convrot file bit for bit with the default MSE clip (`--int8-clip absmax` gives plain absmax scaling). The same 224 int8 layers with group size 256 and per row scales, the same 256 fp8 layers with scalar scales and the full precision flag on the output projections.

LoRA files in the diffusers naming of the Krea 2 conversion (`transformer.transformer_blocks.N.attn.to_q`, `text_fusion`, `img_in`, `time_embed`, `txt_in`, `time_mod_proj`, as written by OneTrainer and ai-toolkit) are mapped to the ComfyUI names automatically, get block shaping, and can be written out in ComfyUI or kohya naming.

## Block shaping

Every LoRA row and the second checkpoint can be shaped across the model's 28 blocks with the Neo-LoraCtl presets: COMPOSITION (blocks 0 to 8), CHARACTER (9 to 18), STYLE (19 to 27), with the modifiers Emphasize, Suppress and Isolate and a contrast slider. FULL, the default, applies no shaping. Two one click recipes come from Neo-LoraCtl's calibration:

- Character LoRA, keep the checkpoint's style: STYLE + Suppress at contrast 0.5.
- Style LoRA, protect faces: CHARACTER + Suppress at contrast 0.5.

Shaping of a checkpoint applies to the second checkpoint only. It scales that checkpoint's contribution per block, which is SuperMerger's block weighted merge. The second checkpoint also has a non block weight for the text side and the projections, which sit outside the block mask.

Timestep scheduling cannot be baked into a file and is not offered.

## Checkpoint merge methods and what the weight means

A is the primary checkpoint and always has weight 1. B is the secondary with weight w. C is the optional reference, the common ancestor of A and B, usually the official Turbo file.

| Method | Formula | Meaning of w |
|---|---|---|
| Add difference (default) | A + w (B - C) | how many times B's change from C is applied to A. 1 transplants B's fine tune onto A, -1 removes it. Without C, C = A |
| Weighted sum | (1 - w) A + w B | position between A (0) and B (1). Outside that range extrapolates |
| SLERP | spherical interpolation | as weighted sum |
| Cosine A, Cosine B | SuperMerger | per column: keep A where A and B agree, take B where they differ. w shifts everything toward B. Use the same storage precision for A and B |
| TIES | trim, elect signs, merge (A - C) and w (B - C), add lambda x result to C | scales B's change before trimming |
| DARE | drop p of each change at random, rescale, sum | as add difference |
| trainDifference | SuperMerger | as add difference, damped where A already moved toward B |
| Extract | SuperMerger | interpolation of the two changes, masked to their common (beta 0) or distinct (beta 1) parts |

## Command line

```bash
run.bat inspect model.safetensors lora.safetensors
```

```bash
run.bat convert krea2_turbo_bf16.safetensors krea2_turbo_int8_convrot.safetensors --format int8_convrot
```

```bash
run.bat lora-merge "epoch08.safetensors|1" "epoch10.safetensors|1" -o merged.safetensors --average
```

```bash
run.bat lora-merge "character.safetensors|1|STYLE:Suppress:0.5" -o character_clean.safetensors
```

```bash
run.bat extract krea2_turbo_bf16.safetensors finetune.safetensors -o finetune_lora.safetensors --rank 32 --analyze
```

```bash
run.bat ckpt-merge -A base.safetensors -B "finetune.safetensors|1|STYLE:Isolate:1.0@0" -C krea2_turbo_bf16.safetensors -o style_only.safetensors --as-lora 32
```

```bash
run.bat run my_recipe.json
```

LoRA and checkpoint arguments take the form `FILE|WEIGHT|SHAPING`, where SHAPING is `PRESET:MODIFIER:CONTRAST[:BOOST][@NONBLOCK]`.

## Layout

- `krea2_merge_tool.py`: entry point.
- `k2merge/`: engine, CLI, GUI.
- `reference/`: the headers of the three official Krea 2 files and a script that summarizes them.
- `fixtures/`: generator of synthetic mini Krea 2 files used by the tests.
- `tests/`: `venv\Scripts\python -m pytest tests`.
- `tools/scale_test.py`: full size synthetic files and memory measurements.
- `tools/real_runs.py`: the phase 8 runs on the real files, writing only into `samples/scratchpad`.
- `working_specs.md`, `development_plan.md`: the specification and the plan.
