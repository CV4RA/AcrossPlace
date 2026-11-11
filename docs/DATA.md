# Data preparation and identity

The initial selected image pool contains 62,279 places and 249,116 images. After generation and formal eligibility filtering, the benchmark contains **56,726 places**, 156,800 rule-valid texts and 113,404 independent target images. “62,279” is not the final training-place count.

Obtain GSV-Cities images under upstream terms. Extract the project's independently supplied `GSVCities_62279_images.tar` into `datasets/GSVCities/`, so that `Images/<city>/<filename>.jpg` is reachable. Obtain the associated formal annotation directory, including its original `manifest.json`, `valid.jsonl` and all files listed in the manifest. The dataset is available from [Google Drive](https://drive.google.com/file/d/13639QpS-GX7ZeibqKaxd45dHxIo5Cf5W/view?usp=drive_link). The repository does not embed the dataset.

```text
datasets/
  GSVCities/Images/<city>/*.jpg
  formal/manifest.json
  formal/valid.jsonl
  formal/{train,val,test}*.jsonl
artifacts/
  siglip2-base-patch16-384/
  blip-itm-base-coco/
  clip-vit-base-patch16/
  clip-vit-base-patch32/
```

Copy formal metadata exactly. **Do not rewrite paths inside JSONL**: their bytes are hashed. `data.canonical_prefix` is `/root/autodl-tmp/datasets/GSVCities`; `data.image_root` points to the actual local image root. The loader maps the former to the latter with containment checks. Run `place-retrieval verify-data --config configs/gsv_full.yaml --check-images` to verify file identity, counts, place separation and target-image hashes.

P0 fingerprint: `0aa126e367f28be0b4f63356081c55dc9c3525c99cd70ba81a5689dd9aa06d4a`.

P1 fingerprint: `fa9e4fe583d0e364883aa4a9629caf287e31d50e4f6a52babbbe476d7e430378`.

`protocols/P0/place_assignments.json` records the final P0 assignment. P1 is allocated deterministically by city metadata using seed 2027, with four test cities and four validation cities chosen to approximate 15% each. Its source formal identity and image/panorama duplicate audit are recorded in the derived manifest. See `python -m place_retrieval.study.protocol --help`.

Source images contribute descriptions; target images form image queries/galleries. A source image is never admitted to the independent target gallery. Relevance is same-place identity, with multiple positive images/texts. Main experiments use rule-valid text; `uncertain` text is retained for diagnostics, never substituted into main validation/test data.

Metadata/coordinates may be used during construction and audit. They are not query inputs to the retrieval models. The configured geographic-negative filter is only active when both rows provide coordinates; formal text rows omit GPS, so this release does not claim an effective geographic-negative intervention in those runs.
