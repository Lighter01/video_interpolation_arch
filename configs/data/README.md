# Stage 1 Data Configs

Data YAML files define pipeline parameters. Runtime roots still come from `.env`; `DATASET_ROOT` is the root used for `sources/...` paths.

## `index_vimeo_triplet.yaml`

Purpose: build a source-level `sequence_index.csv` for the existing Vimeo triplet dataset without copying frames.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `source_dir` | path string | default `sources/vimeo_triplet`, relative to `DATASET_ROOT` unless absolute | Directory containing `sequences/` and split lists | Yes, rebuild index |
| `source_group` | string | default `vimeo` | Written to `source_group` column | Yes, rebuild index |
| `source_dataset` | string | default `vimeo_triplet` | Written to `source_dataset` and `sequence_id` | Yes, rebuild index |
| `train_list` | filename | default `tri_trainlist.txt` | Original train list read under `source_dir` | Yes, rebuild index |
| `test_list` | filename | default `tri_testlist.txt` | Original test list read under `source_dir` | Yes, rebuild index |
| `output_path` | path string or `null` | default `null` | `null` writes `source_dir/sequence_index.csv`; relative paths resolve from repo root; absolute paths are allowed for smoke output | No data change, rerun command to write elsewhere |
| `limit` | integer or `null` | default `null`; positive integer for smoke runs | Caps records written, useful for safe checks | Yes, rebuild index |
| `validate_images` | boolean | default `true` | Checks `im1.png`, `im2.png`, `im3.png` exist and reads image size | Yes, rebuild index |

Input format:

- `train_list` and `test_list` contain relative sequence paths such as `00001/0001`.
- Each listed directory must contain `im1.png`, `im2.png`, and `im3.png`.
- Absolute paths, URIs, and parent-directory paths in list files are rejected.

Output format:

- CSV columns follow the compact `sequence_index.csv` contract.
- Extra column: `original_split` with `train` or `test`.
- `relative_sequence_dir` is relative to `DATASET_ROOT`.

Safe smoke example:

```bash
uv run python -m video_interpolation.cli data index-vimeo-triplets --limit 3 --output /tmp/vimeo_sequence_index.csv
```

Inspect:

```bash
head -5 /tmp/vimeo_sequence_index.csv
wc -l /tmp/vimeo_sequence_index.csv
```

## `global_index.yaml`

Purpose: combine source-level `sequence_index.csv` files into one deterministic global sequence index.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `source_index_globs` | list of strings | default `["sources/*/sequence_index.csv"]`, relative to `DATASET_ROOT` unless absolute | Discovers source index CSV files | Yes, rebuild global index |
| `source_index_paths` | list of paths | default `[]`, relative to `DATASET_ROOT` unless absolute | Adds explicit source indexes; useful for smoke runs | Yes, rebuild global index |
| `selected_source_groups` | list of strings | default `[]` means all groups | Filters records by `source_group` | Yes, rebuild global index |
| `output_path` | path string | default `global_sequence_index.csv`, relative to `DATASET_ROOT` unless absolute | Output CSV path | Rerun command to write elsewhere |
| `limit` | integer or `null` | default `null`; positive integer for smoke runs | Caps written rows after deterministic sorting | Yes, rebuild global index |

Input format:

- Each source index must satisfy the `sequence_index.csv` contract.
- `relative_sequence_dir` values must be relative to `DATASET_ROOT`.
- Extra columns such as Vimeo `original_split` are preserved.

Output format:

- CSV at `DATASET_ROOT/global_sequence_index.csv` by default.
- Required columns match `sequence_index.csv`; extra source columns may be present.
- Rows are sorted by source group, dataset, source video id, sequence id, and sequence directory.

Safe smoke example:

```bash
uv run python -m video_interpolation.cli data build-global-index \
  --config configs/data/global_index.yaml \
  --source-index sources/tmp_test/sequence_index.csv \
  --output /tmp/stage1_dataset_versions/global_sequence_index.csv
```

Inspect:

```bash
head -5 /tmp/stage1_dataset_versions/global_sequence_index.csv
wc -l /tmp/stage1_dataset_versions/global_sequence_index.csv
```

Common failures:

- No source indexes match the config.
- Source indexes miss required columns.
- A source index contains absolute, URI, or parent-directory paths.

## `dataset_version.yaml`

Purpose: build a manifest-only dataset version with `train_all.csv`, `val_all.csv`, `test_all.csv`, and `dataset_config.yaml`.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `dataset_version_id` | string | example `stage1_default` | Output subdirectory name under `output_root` | Yes, rebuild version |
| `source_index_path` | path string | default `global_sequence_index.csv`, relative to `DATASET_ROOT` unless absolute | Global index to read | Yes, rebuild version |
| `output_root` | path string | default `dataset_versions`, relative to repo root unless absolute | Directory that receives dataset-version folders | Rerun command to write elsewhere |
| `split_seed` | integer | default/example `42` | Deterministic source-video split and sampling seed | Yes, rebuild version |
| `split_ratios` | mapping | keys `train`, `val`, `test`; default `0.8/0.1/0.1` | Controls logical split assignment by `source_video_id` | Yes, rebuild version |
| `selected_source_groups` | list of strings | default `[]` means all groups | Filters global index rows before manifest creation | Yes, rebuild version |
| `triplet_policy` | string | `first_triplet`, `center_triplet`, `wide_triplet`, `all_local_triplets`; default `wide_triplet` | Selects triplets from sequences longer than 3 frames; length-3 sequences always use all three frames | Yes, rebuild version |
| `mixing_mode` | string | `none`, `data_accumulation`, `quality_drift`; default `data_accumulation` | Applies train-split old/new pool sampling | Yes, rebuild version |
| `old_source_groups` | list of strings | default/example `["vimeo"]` | Source groups treated as old/base data for mixing | Yes, rebuild version |
| `new_source_groups` | list of strings | default/example `["anime", "small_example"]` | Source groups treated as new/domain data for mixing | Yes, rebuild version |
| `old_min_ratio` | float | default `0.70` | Documented minimum old-data target for `data_accumulation` | Yes, rebuild version |
| `new_max_ratio` | float | default `0.30` | Maximum new-data fraction in `data_accumulation` | Yes, rebuild version |
| `new_min_ratio` | float | default `0.20` | Minimum new-data target in `quality_drift`; oversampling allowed only there | Yes, rebuild version |
| `old_max_ratio` | float | default `0.80` | Maximum old-data fraction in `quality_drift` | Yes, rebuild version |
| `train_budget` | integer or `null` | default `null`; positive integer | Caps train samples after split/mixing; `null` uses available train samples | Yes, rebuild version |
| `limit_sequences` | integer or `null` | default `null`; positive integer for smoke runs | Caps source sequence rows before triplet generation | Yes, rebuild version |
| `validate_frame_paths` | boolean | default `true` | Verifies manifest frame files exist under `DATASET_ROOT` | Yes, rebuild version |
| `frame_name_pattern` | string | `auto`, `im`, `frame`; default `auto` | Chooses `im1.png` style, `frame_000.png` style, or infers from files | Yes, rebuild version |
| `preprocessing` | mapping | free-form documented assumptions | Copied into `dataset_config.yaml` for provenance | Yes, rebuild version |

Input format:

- Reads a global sequence index CSV.
- Sequences from Vimeo usually contain `im1.png`, `im2.png`, `im3.png`.
- Sequences from preprocessing contain `frame_000.png`, `frame_001.png`, `frame_002.png`, and so on.
- Splits are assigned by `source_video_id`; individual triplets from the same source video never cross train/val/test.

Output format:

- `train_all.csv`, `val_all.csv`, and `test_all.csv` use the triplet manifest contract.
- Frame paths are relative to `DATASET_ROOT`.
- `dataset_config.yaml` records version id, source index reference, split policy, selected groups, triplet policy, mixing policy, preprocessing assumptions, counts, and timestamp.
- In tiny versions where the train split contains only old or only new rows, mixing keeps the available train rows instead of creating an empty train manifest.

Safe smoke example:

```bash
uv run python -m video_interpolation.cli data build-dataset-version \
  --config configs/data/dataset_version.yaml \
  --source-index /tmp/stage1_dataset_versions/global_sequence_index.csv \
  --output-root /tmp/stage1_dataset_versions \
  --dataset-version-id smoke_version \
  --limit-sequences 10
```

Inspect:

```bash
find /tmp/stage1_dataset_versions/smoke_version -maxdepth 1 -type f -print
head -5 /tmp/stage1_dataset_versions/smoke_version/train_all.csv
cat /tmp/stage1_dataset_versions/smoke_version/dataset_config.yaml
```

Common failures:

- The global index has no rows after source-group filtering.
- `wide_triplet` is requested for an even `sequence_length`.
- `validate_frame_paths: true` and the frame files cannot be found under `DATASET_ROOT`.
- `frame_name_pattern: auto` cannot infer either `im*.png` or `frame_*.png`.
- Split ratios are all zero or contain negative values.

## `preprocess_anime.yaml`

Purpose: sample raw videos into PNG sequences under `DATASET_ROOT/sources/anime/sequences/` and write `DATASET_ROOT/sources/anime/sequence_index.csv`.

`preprocess_test.yaml` uses the same fields but points at `raw_data/tmp_test` and `sources/tmp_test` for MP4/MKV diagnostics.

Fields:

| Field | Type | Allowed/default | Effect | Rerun needed |
| --- | --- | --- | --- | --- |
| `raw_input_dir` | path string | default `raw_data/anime`, relative to repo root unless absolute | Input video tree scanned recursively | Yes, rerun preprocessing |
| `output_source_dir` | path string | default `sources/anime`, relative to `DATASET_ROOT` | Output source directory | Yes, rerun preprocessing |
| `source_group` | string | default `anime` | Written to `source_group` | Yes, rerun preprocessing |
| `source_dataset` | string | default `anime` | Written to `source_dataset` and `sequence_id` | Yes, rerun preprocessing |
| `sequence_length` | integer | default `3`, minimum `3` | Number of frames per sampled sequence | Yes, rerun preprocessing |
| `max_frame_step` | integer | default `2`, allowed `0`, `1`, `2`; must be `0` when `sequence_length > 3` | Controls temporal stride as `frame_step + 1`; `0` means adjacent frames | Yes, rerun preprocessing |
| `min_sequences_per_video` | integer | default `1`, minimum `0` | Lower target quota before candidate availability is applied | Yes, rerun preprocessing |
| `max_sequences_per_video` | integer | default `50`, must be >= minimum | Upper target quota per video | Yes, rerun preprocessing |
| `quota_scale` | float | default `1.0`, minimum `0` | Scales candidate count before min/max caps | Yes, rerun preprocessing |
| `random_seed` | integer | default/example `20260522` | Makes candidate selection deterministic | Yes, rerun preprocessing |
| `resize` | string or `null` | default `null`; format `WIDTHxHEIGHT` | Resizes extracted frames before writing PNGs | Yes, rerun preprocessing |
| `static_ssim_threshold` | float | default `0.95` | Rejects triplets when either adjacent SSIM score is at least this value | Yes, rerun preprocessing |
| `reject_static` | boolean | default `true` | Enables/disables static-triplet rejection | Yes, rerun preprocessing |
| `limit_videos` | integer or `null` | default `null`; positive integer for smoke runs | Caps number of discovered input videos | Yes, rerun preprocessing |
| `only_video` | string or `null` | default `null`; filename or path relative to `raw_input_dir` | Selects one video for debugging | Yes, rerun preprocessing |
| `video_glob` | string or `null` | default `null`; glob matching filename or relative path | Selects matching videos for debugging | Yes, rerun preprocessing |
| `max_duration_sec` | float or `null` | default `null`; positive seconds | Limits each video to the first N seconds for smoke/debug runs | Yes, rerun preprocessing |
| `max_frames` | integer or `null` | default `null`; positive frame count | Limits each video to the first N frames for smoke/debug runs | Yes, rerun preprocessing |
| `decode_strategy` | string | default `auto`; allowed `auto`, `sequential`, `segment_seek` | Controls selected-frame extraction. `auto` uses segment seek for later frames and sequential decoding for short early ranges | Yes, rerun preprocessing |

Input format:

- Supported video extensions: `.mkv`, `.mov`, `.mp4`, `.webm`.
- Videos are discovered recursively under `raw_input_dir`.

Output format:

- PNG sequences under `DATASET_ROOT/<output_source_dir>/sequences/<source_video_id>/<sequence_number>/`.
- Frames are named `frame_000.png`, `frame_001.png`, and so on.
- Source index at `DATASET_ROOT/<output_source_dir>/sequence_index.csv`.
- Index paths are relative to `DATASET_ROOT`.

Side effects and safe reruns:

- The command creates output directories as needed.
- It rewrites `sequence_index.csv` atomically after processing.
- Existing sequence directories for the same source video/sequence number may be overwritten frame-by-frame. For experiments, use a separate `output_source_dir` or remove old generated outputs intentionally before rerunning.
- Use `--limit-videos 1` for a small first run.
- `--limit-videos` limits the number of files, not the duration of each file. Use `--max-duration-sec` or `--max-frames` for long episodes.

Debug selectors and limits:

```bash
uv run python -m video_interpolation.cli data preprocess-videos \
  --config configs/data/preprocess_test.yaml \
  --only-video one_piece_test_1m.mkv \
  --max-duration-sec 10 \
  --debug-progress
```

Selected-frame decoding:

- `sequential` decodes from the start of the video until all requested frames are recovered.
- `segment_seek` groups requested frames into compact intervals, seeks near each interval, and decodes until the interval end.
- `auto` uses `segment_seek` when requested frames are later in the video and falls back to sequential decoding if seeking misses frames.

Inspect:

```bash
find datasets/sources/anime/sequences -maxdepth 3 -type f | head
head -5 datasets/sources/anime/sequence_index.csv
```

Common failures:

- Missing `raw_input_dir`.
- Invalid `resize` format.
- `max_frame_step` outside `0`, `1`, `2`.
- `sequence_length > 3` with nonzero `max_frame_step`.
- Videos with unsupported codecs or unreadable metadata.
- FFmpeg/PyAV may print audio-stream warnings during container probing. The preprocessing code selects and decodes only the video stream; audio warnings are non-blocking unless the command records a per-video failure.
