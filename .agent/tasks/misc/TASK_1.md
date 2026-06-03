# TASK — Add Slow-Motion Output Playback Mode to Video Inference

## Purpose

Add a second output playback mode to the existing video inference pipeline.

The current inference behavior writes interpolated frames into a video with increased FPS, preserving the original video duration:

```text
input_fps = fps
interpolation_factor = N
output_fps = fps * N
duration ~= original duration
```

The new slow-motion output mode should write the same interpolated frame sequence at the original FPS, increasing playback duration:

```text
input_fps = fps
interpolation_factor = N
output_fps = fps
duration ~= original duration * N
```

This feature is meant for demos and user-facing output options. It should be available through:

- core video inference config;
- CLI inference commands;
- `PracticalRIFEVideoInferenceRunner`;
- BentoML example services.

Do not change model inference logic, model adapters, ONNX/PyTorch runtime logic, or frame interpolation semantics.

---

## Current context

The current video inference implementation lives mainly in:

```text
src/video_interpolation/inference.py
```

The current pipeline:

- reads input video frames;
- generates intermediate frames;
- writes source and generated frames to an output video;
- currently computes output FPS as:

```text
output_fps = input_fps * interpolation_factor
```

- currently may preserve/remux audio streams into the output video.

The new feature should only change output playback timing and audio handling.

---

## Main design decision

Add a new playback-mode config field.

Preferred name:

```python
output_playback_mode
```

Preferred enum:

```python
class VideoOutputPlaybackMode(StrEnum):
    REAL_TIME = "real_time"
    SLOW_MOTION = "slow_motion"
```

The enum name is a bit long, but acceptable and explicit.

The BentoML service interface should expose this as a simple string argument:

```python
output_playback_mode: str = "real_time"
```

Allowed values:

```text
real_time
slow_motion
```

Default:

```text
real_time
```

---

## Do not use legacy output_fps_multiplier

`output_fps_multiplier` is a legacy field.

Do not build this feature around it.

Do not rely on it for the new behavior.

Do not remove it in this task, to avoid unnecessary compatibility risk.

The new `output_playback_mode` should be the authoritative field for choosing output FPS behavior.

---

## Output FPS behavior

Implement one central helper or equivalent logic:

```python
def resolve_output_fps(
    input_fps: float,
    interpolation_factor: int,
    output_playback_mode: VideoOutputPlaybackMode,
) -> float:
    ...
```

Required behavior:

```text
real_time:
  output_fps = input_fps * interpolation_factor

slow_motion:
  output_fps = input_fps
```

Use the resolved `output_fps` everywhere the encoder/output stream needs the FPS.

Do not change the number or order of frames written.

The interpolation pipeline should still write:

```text
source frame 0
generated frames between source frame 0 and source frame 1
source frame 1
generated frames between source frame 1 and source frame 2
source frame 2
...
```

The only difference is the output stream FPS.

---

## Audio behavior

In `real_time` mode, preserve the current audio behavior.

In `slow_motion` mode:

- do not preserve/remux audio;
- do not create output audio streams;
- do not call audio remux/copy logic;
- return/report `audio_streams_preserved = 0`.

Do not implement audio stretching/time-scaling.

Do not add `audio_disabled_reason`; this is unnecessary.

Rationale:

- slow-motion output changes video duration;
- original audio would no longer match the video;
- audio time-stretching would add unnecessary complexity for this demo feature.

---

## Config integration

Add `output_playback_mode` to `VideoInferenceConfig`.

Requirements:

- default to `real_time`;
- parse from config mappings;
- validate allowed values;
- provide a resolver method if consistent with current config style, e.g.:

```python
resolved_output_playback_mode()
```

Update inference configs only where appropriate.

Do not create separate configs just for slow motion unless a small demo config is clearly useful.

---

## CLI integration

Add CLI support to video inference commands.

Preferred flag:

```text
--output-playback-mode real_time
--output-playback-mode slow_motion
```

Do not use a boolean `--slow-motion` flag unless that fits the existing CLI style better. The enum/string flag is preferred because it is more extensible.

Apply the flag to relevant local video inference commands, especially Practical-RIFE commands used for demos.

Existing behavior must remain unchanged when the flag is omitted.

---

## Serving and BentoML integration

Update `PracticalRIFEVideoInferenceRunner` so it accepts and passes through:

```python
output_playback_mode: str = "real_time"
```

Update BentoML example services so `interpolate_video(...)` accepts:

```python
output_playback_mode: str = "real_time"
```

and passes it into the runner.

The service should not implement playback logic itself.

The service should only pass the parameter to the existing inference interface.

Returned service dictionaries should include the selected mode and resolved output FPS if the current response structure supports it, for example:

```python
{
    "output_playback_mode": "slow_motion",
    "input_fps": ...,
    "output_fps": ...,
    ...
}
```

Do not add `audio_disabled_reason`.

---

## Result/metadata integration

If useful, add `output_playback_mode` to `VideoInferenceResult`.

At minimum, ensure that result/report/logging can distinguish:

```text
real_time
slow_motion
```

MLflow/logging/benchmark metadata should include:

```text
output_playback_mode
input_fps
output_fps
interpolation_factor
audio_streams_available
audio_streams_preserved
```

For slow-motion mode, `audio_streams_preserved` should be `0`.

---

## Benchmark integration

If video benchmark code uses `VideoInferenceConfig`, allow it to pass through `output_playback_mode`.

Do not make slow motion the default benchmark behavior.

Benchmark default should remain `real_time`.

If benchmark reports include output FPS or audio metrics, make sure they remain correct for both modes.

---

## Tests

Add focused tests.

Required test cases:

1. Output FPS resolution:
   - `real_time`: `output_fps = input_fps * interpolation_factor`;
   - `slow_motion`: `output_fps = input_fps`.

2. Config validation:
   - accepts `real_time`;
   - accepts `slow_motion`;
   - rejects invalid mode.

3. Inference result metadata:
   - includes output playback mode if result was extended;
   - reports expected output FPS.

4. Audio behavior:
   - `real_time` preserves existing audio logic;
   - `slow_motion` does not attempt audio remux;
   - `slow_motion` reports `audio_streams_preserved = 0`.

5. CLI parsing:
   - default is `real_time`;
   - `--output-playback-mode slow_motion` reaches `VideoInferenceConfig`.

6. BentoML/serving wrapper:
   - accepts `output_playback_mode`;
   - passes it to the inference config;
   - default remains `real_time`.

Use fake/lightweight tests where possible.

Do not require CUDA for unit tests.

If adding smoke tests, keep them short and bounded.

---

## Documentation

Update human-facing documentation.

Document:

- what `output_playback_mode` means;
- difference between `real_time` and `slow_motion`;
- that the same generated frames are written in both modes;
- that `real_time` increases FPS and preserves approximate original duration;
- that `slow_motion` keeps original FPS and increases duration;
- that slow-motion output does not preserve audio;
- CLI usage examples;
- BentoML request parameter usage;
- default value: `real_time`.

Update:

- Stage 2 inference/runtime docs;
- BentoML example docs if present;
- config README files if config fields are changed;
- `.agent/docs/PROJECT_MAP.md` if files or public interfaces change;
- active ExecPlan if this task is tracked there.

---

## Constraints

Do not implement:

- audio time-stretching;
- scene-cut detection;
- batch-inference changes;
- quality-evaluation changes;
- ONNX export changes;
- model runtime changes;
- new model adapters;
- FastAPI/Celery/MinIO/frontend logic.

Do not remove `output_fps_multiplier` in this task.

Do not change model weights.

Do not change interpolation frame ordering.

---

## Validation

Run:

```bash
uv run pytest
uv run ruff check src tests
```

If the uv cache is not writable:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src tests
```

Run only safe smoke checks.

Do not run long video jobs.

---

## End-of-task summary

At the end, summarize:

- what playback mode enum/config field was added;
- how output FPS is resolved in each mode;
- how slow-motion audio behavior is handled;
- what CLI flags were added;
- how BentoML examples were updated;
- what tests were added;
- what validation commands were run;
- what docs/project-map/ExecPlan updates were made;
- any limitations or follow-up work.
