# Data, models and attribution

AcrossPlace builds on **GSV-Cities**. Street-view photographs originate from that dataset and its upstream imagery providers. The 44 unmodified photographs in `website/dist/assets/` are attributed research illustrations; `provenance.json` records their dataset-relative names and SHA256 hashes. They are **not relicensed under the software MIT license**. The complete imagery archive is not distributed in this repository. Obtain and use upstream imagery under its applicable terms, including restrictions on redistribution. A public repository does not grant additional image rights.

Pretrained models are downloaded separately from Google (SigLIP2), Salesforce (BLIP), OpenAI (CLIP), and Qwen (Qwen3-VL); their model cards and licenses govern their weights and use. Hugging Face Transformers, PEFT and PyTorch retain their respective licenses. BLIP+MS is an adapted LaVPR-style baseline under a common benchmark budget, not a new method introduced by this release.

The hosted manuscript and supplement are research drafts, not a claim of conference acceptance. CVPR style files are not redistributed as project-owned code. Internal names `AcrossViews`, `T4`, `B0`, and `siglip2-place-lora-v1` may appear in historical artifact formats; the public benchmark name is **AcrossPlace**.
