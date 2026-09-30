# Live Traffic Intelligence

One script, one loop, live-capable. Replaces the 12 offline notebook phases with a single streaming pipeline.

```
frame -> YOLOv8 -> DeepSORT/ByteTrack -> scale-free analytics -> TrafficNet (CNN-LSTM) -> video overlay + terminal
```

## Setup and run

```bash
pip install -r requirements.txt

python live_traffic.py --source highway.mp4                  # recorded video
python live_traffic.py --source highway.mp4 --realtime       # replay at native speed (live simulation)
python live_traffic.py --source 0                            # webcam
python live_traffic.py --source rtsp://user:pw@ip:554/stream --tracker bytetrack   # IP camera
```

Press `q` or `Esc` to quit. Useful flags: `--imgsz 960` (aerial / far cameras), `--tracker bytetrack`
(faster), `--save out.mp4`, `--no-show` (headless), `--capacity N` and `--free-flow-kmh V` (skip auto-calibration).

## Outputs

On video: boxes, track IDs, per-vehicle km/h, motion trails (red = stopped), HUD with vehicles now (by class),
total unique vehicles, flow (veh/min), density level and score, average speed, queue, +30 s forecast,
clearance time, incident flag, FPS and latency. The same numbers print once per second in the terminal.

## Low latency

- Live sources are read by a separate thread that keeps only the newest frame, so lag never accumulates.
- FP16 on GPU, frames downscaled to `--max-width` (1280 default), YOLOv8n by default.
- The LSTM runs once per second on a 30-step window, so its cost is negligible.
- Latency of each stage is shown on screen. If FPS is low: `--tracker bytetrack`, smaller `--imgsz`, or a GPU.

## Angle independence (how it is approached)

- Speed is measured in body-lengths per second, using the vehicle's own bounding box as the ruler, so perspective and zoom largely cancel. It is converted to km/h with class-typical sizes.
- Density, queue and speed features are ratios against the scene's own adaptive baseline (running percentiles), not raw pixels or raw counts.
- Queue = stopped vehicles (direction-agnostic). Clearance = queue / discharge rate.
- TrafficNet only sees these ratios, so it can be trained on many camera angles together.

## Training TrafficNet (optional; a heuristic forecaster is used until you do)

```bash
# 1) log 1 Hz features from each recorded video (different angles!)
python live_traffic.py --source v1.mp4 --no-show --log logs/v1.csv
python live_traffic.py --source v2.mp4 --no-show --log logs/v2.csv
python live_traffic.py --source v3.mp4 --no-show --log logs/v3.csv
# 2) train (last CSV is held out as an unseen-angle check)
python trafficnet.py --csv logs/*.csv --out trafficnet.pt
# 3) run with it
python live_traffic.py --source 0 --model trafficnet.pt
```

Labels are self-supervised from the log (future density, lookahead clearance, rule incidents): no manual annotation.

## Honest limitations

- Angle independence is reduced, not solved. Vehicles driving straight toward or away from the camera are under-measured in speed (bbox grows/shrinks rather than moving). Camera shake is not compensated. Prove it on unseen angles using the held-out video.
- km/h is approximate (class-typical sizes). For calibrated speed, use a homography from known road markings.
- Auto-calibration learns "busy" and "free-flow" from the scene's history. A road that is jammed for its whole first 10 minutes will be mis-calibrated; pass `--capacity` / `--free-flow-kmh`.
- Clearance labels are pseudo-labels (as in your report), so R² will be modest. Treat the value as an estimate, and it shows "not clearing" when there is no discharge.
- Total unique vehicles can be inflated by ID switches, especially with ByteTrack in dense occlusion.
- The pipeline was tested here with simulated detections (jam onset, incident, recovery) and rendering. YOLO, DeepSORT and the PyTorch model were not run in this sandbox (no internet), so do a first run on your machine and tell me any error.
