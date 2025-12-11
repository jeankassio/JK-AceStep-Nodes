# JK AceStep Nodes for ComfyUI

Advanced nodes optimized for [ACE-Step](https://huggingface.co/Accusoft/ACE-Step) audio generation in ComfyUI.

## 📦 Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/jeankassio/JK-AceStep-Nodes.git
```

Restart ComfyUI. Nodes will appear under `JK AceStep Nodes/` categories.

---

## 🎵 Nodes

### Ace-Step KSampler (Basic)
Full-featured sampler with quality check, advanced guidance (APG, CFG++, Dynamic CFG), anti-autotune smoothing, and automatic step optimization.

**Category:** `JK AceStep Nodes/Sampling`

---

### Ace-Step KSampler (Advanced)
Extended sampler with start/end step control for multi-pass workflows and refinement.

**Category:** `JK AceStep Nodes/Sampling`

---

### Ace-Step Prompt Gen
Prompt generator with **150+ professional music styles** (Electronic, Rock, Jazz, Classical, Brazilian, World Music, and more).

**Category:** `JK AceStep Nodes/Prompt`

---

## 🎤 Lyrics Generators

Ten AI-powered lyrics generation nodes supporting various LLM providers:

### Ace-Step OpenAI Lyrics
Lyrics generation using OpenAI GPT models.

**Supported Models (December 2025):**
- `gpt-5.1` - Reasoning model (latest)
- `gpt-5.1-codex` - Coding optimized
- `gpt-5` - High performance
- `gpt-5-pro` - Professional variant
- `gpt-4o` - Multimodal (recommended)
- `gpt-4o-mini` - Fast variant
- `gpt-4-turbo` - High performance
- `gpt-4` - Stable base
- `o3` - Reasoning model
- `o3-mini` - Compact reasoning
- `o1` - Advanced reasoning
- `o1-mini` - Compact advanced reasoning

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Claude Lyrics
Lyrics generation using Anthropic Claude models.

**Supported Models (December 2025):**
- `claude-opus-4.5` - Latest flagship (recommended)
- `claude-opus-4.1` - Previous flagship
- `claude-sonnet-4.5` - Latest balanced
- `claude-sonnet-4` - Previous balanced
- `claude-haiku-4.5` - Latest fast
- `claude-haiku-3.5` - Previous fast
- `claude-3-5-sonnet-20241022` - Snapshot variant
- `claude-3-5-haiku-20241022` - Snapshot variant
- `claude-3-opus-20250219` - Dated variant

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Gemini Lyrics
Lyrics generation using Google Gemini API.

**Supported Models (December 2025):**
- `gemini-3-pro` - Latest pro model (recommended)
- `gemini-2.5-flash` - Fast with latest capabilities
- `gemini-2.5-flash-lite` - Ultra-fast variant
- `gemini-2.5-pro` - High quality
- `gemini-2.0-flash` - Previous generation
- `gemini-2.0-flash-lite` - Previous generation lite

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Groq Lyrics
High-speed lyrics generation using Groq API.

**Supported Models (December 2025):**
- `llama-3.3-70b-versatile` - Meta Llama 3.3 70B (best quality)
- `llama-3.1-8b-instant` - Meta Llama 3.1 8B (fast)
- `llama-guard-4-12b` - Meta Guard model
- `deepseek-v3` - DeepSeek V3
- `mistral-small-3` - Mistral Small v3
- `gpt-oss-120b` - OpenAI OSS 120B
- `gpt-oss-20b` - OpenAI OSS 20B
- Plus additional production models

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Perplexity Lyrics
Lyrics generation using Perplexity Sonar models.

**Supported Models (December 2025):**
- `sonar` - Standard model
- `sonar-pro` - Professional variant
- `sonar-reasoning` - Reasoning-focused
- `sonar-reasoning-pro` - Advanced reasoning
- `sonar-deep-research` - Deep research variant

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Cohere Lyrics
Lyrics generation using Cohere Command models.

**Supported Models (December 2025):**
- `command-a-03-2025` - Latest Command A
- `command-r7b-12-2024` - December 2024 variant
- `command-r-plus-08-2024` - R+ August 2024
- `command-r-08-2024` - R August 2024
- `command-a-translate` - Translation specialist
- `command-a-reasoning` - Reasoning-focused
- `command-a-vision` - Vision capabilities
- `aya-expanse-32b` - Aya Expanse 32B
- `aya-expanse-8b` - Aya Expanse 8B
- `aya-vision` - Aya with vision
- `aya-translate` - Aya translation specialist

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Replicate Lyrics
Lyrics generation using Replicate API models.

**Supported Models (December 2025):**
- `meta/llama-3.1-405b-instruct` - 405B instruction-tuned
- `meta/llama-3.1-70b-instruct` - 70B instruction-tuned
- `meta/llama-3.1-8b-instruct` - 8B instruction-tuned
- `meta/llama-3-70b-instruct` - Llama 3 70B
- `meta/llama-2-70b-chat` - Llama 2 chat 70B
- `mistralai/mistral-7b-instruct-v0.3` - Mistral 7B v0.3
- `mistralai/mistral-small-24b-instruct-2501` - Mistral Small 24B
- `mistralai/mixtral-8x7b-instruct-v0.1` - Mixtral MoE

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step HuggingFace Lyrics
Lyrics generation using HuggingFace Inference API.

**Supported Models (December 2025):**
- `meta-llama/Llama-3.1-405B-Instruct` - Large instruction-tuned
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` - Llama 3.3 70B turbo
- `meta-llama/Llama-3.1-70B-Instruct` - 70B instruction-tuned
- `mistralai/Mistral-Large` - Large Mistral variant
- `microsoft/Phi-4` - Phi-4 model
- `deepseek-ai/deepseek-v3` - DeepSeek V3
- `Qwen/Qwen2.5-72B-Instruct` - Qwen 2.5 72B
- `google/gemma-2-27b` - Gemma 2 27B
- `tiiuae/falcon-180b` - Falcon 180B

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Together AI Lyrics
Lyrics generation using Together AI serverless models.

**Supported Models (December 2025):**
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` - Llama 3.3 70B turbo
- `meta-llama/Llama-3.1-405B-Instruct-Turbo` - Llama 3.1 405B turbo
- `mistralai/Mistral-Small-24B-Instruct-2501` - Mistral Small 24B
- `Qwen/Qwen2.5-72B-Instruct` - Qwen 2.5 72B
- `deepseek-ai/DeepSeek-V3` - DeepSeek V3
- `moonshotai/Kimi-K2-Instruct` - Kimi K2
- `GLM-4-Plus` - GLM 4 Plus
- `Nous-Hermes-3-70B` - Nous Hermes 3 70B
- Plus 100+ additional models available

**Category:** `JK AceStep Nodes/Lyrics`

### Ace-Step Fireworks Lyrics
Lyrics generation using Fireworks AI models (100+ available).

**Supported Models (December 2025):**
- `deepseek-ai/deepseek-v3` - DeepSeek V3 (latest)
- `deepseek-ai/deepseek-r1` - DeepSeek R1 reasoning
- `Qwen/Qwen3-235B-A22B-Instruct` - Qwen 3 235B
- `Qwen/Qwen2.5-72B-Instruct-Turbo` - Qwen 2.5 72B turbo
- `meta-llama/Llama-4-Maverick-17B` - Llama 4 Maverick
- `meta-llama/Llama-4-Scout-17B` - Llama 4 Scout
- `meta-llama/Llama-3.3-70B-Instruct` - Llama 3.3 70B
- `meta-llama/Llama-3.1-405B-Instruct` - Llama 3.1 405B
- `mistralai/Mistral-Large-3-675B-Instruct` - Mistral Large 675B
- `mistralai/Mistral-Small-24B-Instruct-2501` - Mistral Small 24B
- `google/GLM-4.6` - GLM 4.6
- `moonshotai/Kimi-K2` - Kimi K2
- `google/Gemma-3-27b` - Gemma 3 27B
- Plus 90+ additional models available

**Category:** `JK AceStep Nodes/Lyrics`

---

### Ace-Step Save Text
Text saver with auto-incremented filenames and folder support. Works with any lyrics generator.

**Category:** `JK AceStep Nodes/IO`

---

## 🎨 JKASS Custom Sampler

**J**ust **K**eep **A**udio **S**ampling **S**imple - custom sampler optimized for audio generation.

### Available Variants

- **`jkass_quality`** - Second-order Heun method for maximum audio quality
  - Superior accuracy and detail preservation
  - Recommended for final renders
  - ~2x slower than fast variant

- **`jkass_fast`** - First-order Euler method for faster generation
  - Optimized for speed with vectorized operations
  - Good quality with reduced computation time
  - Best for iterations and previews

### Key Features
- No noise normalization (preserves audio dynamics)
- Clean sampling path (prevents word cutting/stuttering)
- Optimized for long-form audio

Select your preferred variant from any sampler dropdown (default: `jkass_quality`).

---

## 📊 Recommended Settings

- **Sampler:** `jkass_quality` (for best quality) or `jkass_fast` (for speed)
- **Scheduler:** `sgm_uniform`
- **Steps:** 80-100
- **CFG:** 4.0-4.5
- **Anti-Autotune:** 0.25-0.35 (vocals), 0.0-0.15 (instruments)

### ✅ Extra settings for reducing 'AI' female vocals
- **Sampler:** `jkass_quality` (for best quality) or `jkass_fast` (for speed)
- **Frequency Damping:** 0.15-0.5 for female vocals to reduce metallic sizzle (0=disabled)
- **Temporal Smoothing:** 0.02-0.12 to reduce pitch quantization and temporal discontinuities
- **Beat Stability:** 0.05-0.2 to preserve stable rhythmic strikes and avoid per-frame jitter
- **Anti-Autotune:** 0.25-0.35 (vocals), 0.0-0.15 (instruments)

---

## 🎯 Quality Check Feature

Automatically tests multiple step counts to find optimal settings for your prompt.

**Important:** Quality scores are **comparative only**. Compare within same prompt/style. Electronic music naturally scores lower than acoustic (both can be excellent quality).

---

## 🔧 Troubleshooting

- **Word cutting/stuttering:** Use `jkass_quality` sampler, disable advanced optimizations
- **Metallic voice:** Increase `anti_autotune_strength` to 0.3-0.4
- **AI-sounding female voice:** Try the following sequence:
  1. Use `jkass_quality` and 80-120 steps, CFG 4.0-4.5, APG enabled
  2. Set Anti-Autotune (0.25-0.35), Frequency Damping (0.15-0.4), Temporal Smoothing (0.02-0.06)
  3. Use the Prompt Gen with `voice_style` -> `natural_female` and add 'breathy, micro pitch variation' in extra prompt
  4. Decode using a high-quality VAE/vocoder (HiFi-GAN, or validated VAE) for improved timbre
  5. If still metallic: de-esser and mild EQ cut at 7-12 kHz; add subtle formant correction and breath overlay
  
  **Optional:** Use the `Ace-Step Post Process` node to apply a quick de-essing (reduce 6-10 kHz energy), spectral smoothing, and subtle breath overlay to humanize the vocal further.
- **Poor quality:** Increase steps (80-120), use CFG 4.0-4.5, enable APG, try `jkass_quality` sampler

---

## 📁 Project Structure

```
JK-AceStep-Nodes/
├── __init__.py                    # Central node aggregator
├── ace_step_ksampler.py           # Samplers (Basic + Advanced)
├── ace_step_prompt_gen.py         # Prompt generator (150+ styles)
├── lyrics_nodes.py                # 10 lyrics generators consolidated
├── ace_step_save_text.py          # Text saver node
├── requirements.txt
└── py/
    └── jkass_sampler.py           # Custom audio sampler
```

### Available Lyrics Generators

- **OpenAI** - gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo, and more
- **Anthropic Claude** - Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
- **Google Gemini** - gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro/flash
- **Groq** - Llama 3.3 70B, Llama 3.1 8B, Llama Guard 3, GPT-OSS (120B/20B), and Llama 4 preview models
- **Perplexity** - Sonar, Sonar Pro, Sonar Reasoning (with 128k context)
- **Cohere** - Command A/R+ (with reasoning & vision), Aya (multilingual)
- **Replicate** - Llama 3.1 (405B/70B/8B), Mistral Small/Nemo, Mixtral
- **HuggingFace** - Llama 3.1, Mistral, DeepSeek, Qwen, Falcon, and 100+ more
- **Together AI** - Llama 3.3/3.1, DeepSeek, Qwen 3, Mistral variants, and 50+ more
- **Fireworks AI** - DeepSeek V3/R1, Qwen 3, Llama 3.3/3.1, Mistral Large/Small, GLM, and 90+ more

---

## 🗣️ How to Use the Vocoder (ADaMoSHiFiGAN)

To enable audio conversion with the vocoder (for improved final audio quality):

1. **Obtain the vocoder files:**
   - `diffusion_pytorch_model.safetensors` (vocoder model)
   - `config.json` (vocoder configuration)

2. **Place both files in the folder:**
   - `JK-AceStep-Nodes/vocoder/`

   The final path should be:
   ```
   JK-AceStep-Nodes/vocoder/diffusion_pytorch_model.safetensors
   JK-AceStep-Nodes/vocoder/config.json
   ```

3. **Done!**
   - The system will automatically detect these files when using nodes with vocoder enabled.
   - If the files are not present, audio will be generated without the vocoder.

> **Tip:**
> Always use the correct file pair (model + config) to avoid artifacts or loading errors.

---

## 📄 License

MIT License

---

**Version:** 2.3  
**Last Updated:** December 2025

🎵 Happy music generation!
