# AcrossPlace project website

Buildless HTML/CSS/ES modules. No account, GPU, backend or package installation is required to browse the recorded examples.

```bash
python -m http.server 4173 --directory dist --bind 127.0.0.1
node --test tests/data.test.js
```

Features: original-image source/target explorer; P0/P1, coarse/reranked and Recall@K controls; real gallery-scaling chart with numerical data; T2I/I2T prediction replay; reproducible-code links. `dist/data.json` is derived from the release evidence and curated, documented case selection. Imagery is unmodified; `dist/assets/provenance.json` lists SHA256 hashes and upstream filenames.

The public instance is https://acrossplace.ornatesparrow2.chatgpt.site. `.openai/hosting.json` identifies that deployment. When forking, register your own Site and replace its project identity before publishing; do not reuse this project's deployment identity. The static `dist/` directory can also be hosted on GitHub Pages without a build step.
