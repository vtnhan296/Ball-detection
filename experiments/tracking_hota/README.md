# YOLO + ByteTrack HOTA Experiment

This experiment evaluates tracking for YOLO + ByteTrack only.

It does not use `team_classifier`, PRTReID, team labels, or role/team clustering.

## Flow

1. Run YOLO tracking with Ultralytics ByteTrack on a full SoccerNet sequence.
2. Save predictions as MOT-style tracking text.
3. Filter SoccerNet ground truth to:
   - `player`
   - `goalkeeper`
   - `referee`
4. Convert each class into a TrackEval MOTChallenge-compatible subset.
5. Run TrackEval HOTA and save `metrics.csv` / `metrics.json`.

## Notebook

Open and run:

```text
experiments/tracking_hota/hota_yolo_bytetrack.ipynb
```

Default sequence:

```python
SEQUENCE = "SNMOT-060"
MAX_FRAMES = None
```

`MAX_FRAMES = None` means the full sequence is evaluated.

## Dependency

TrackEval is required for HOTA:

```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```
