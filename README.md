# JK AceStep Nodes for ComfyUI

Advanced nodes optimized for [ACE-Step](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) audio generation in ComfyUI.

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

### Ace-Step Gemini Lyrics
Lightweight lyric generator using Google Gemini API.

**Category:** `JK AceStep Nodes/Gemini`

---

### Ace-Step Save Text
Text saver with auto-incremented filenames and folder support.

**Category:** `JK AceStep Nodes/IO`

---

## 🎨 JKASS Custom Sampler

**J**ust **K**eep **A**udio **S**ampling **S**imple - custom sampler optimized for audio generation.

- No noise normalization (preserves audio dynamics)
- Clean sampling path (prevents word cutting/stuttering)
- Optimized for long-form audio

Select `jkass` from any sampler dropdown.

---

## 📊 Recommended Settings

- **Sampler:** `jkass`
- **Scheduler:** `sgm_uniform`
- **Steps:** 80-100
- **CFG:** 4.0-4.5
- **Anti-Autotune:** 0.25-0.35 (vocals), 0.0-0.15 (instruments)

---

## 🎯 Quality Check Feature

Automatically tests multiple step counts to find optimal settings for your prompt.

**Important:** Quality scores are **comparative only**. Compare within same prompt/style. Electronic music naturally scores lower than acoustic (both can be excellent quality).

---

## 🔧 Troubleshooting

- **Word cutting/stuttering:** Use `jkass` sampler, disable advanced optimizations
- **Metallic voice:** Increase `anti_autotune_strength` to 0.3-0.4
- **Poor quality:** Increase steps (80-120), use CFG 4.0-4.5, enable APG

---

## 📁 Project Structure

```
JK-AceStep-Nodes/
├── __init__.py
├── ace_step_ksampler.py          # Samplers (Basic + Advanced)
├── ace_step_prompt_gen.py        # Prompt generator (150+ styles)
├── gemini_nodes.py                # Gemini lyrics + text saver
├── requirements.txt
└── py/
    └── jkass_sampler.py          # Custom audio sampler
```

---

## 📄 License

MIT License

---

**Version:** 2.2  
**Last Updated:** December 2025

🎵 Happy music generation!
