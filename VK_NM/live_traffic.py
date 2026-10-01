#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')

"""
Live Traffic Intelligence  -  one script, one loop, works on files, webcams and RTSP streams.

  frame -> YOLOv8 (vehicles) -> DeepSORT / ByteTrack (IDs) -> scale-free scene analytics
        -> TrafficNet CNN-LSTM (forecast, clearance, incident)  -> overlay on video + terminal line

Angle independence comes from never using raw pixels as a unit:
  * speed is measured in "vehicle body lengths per second" (bbox size is the local ruler,
    so perspective/zoom cancels out), then converted to km/h with class-typical sizes
  * density = count / adaptive capacity, box coverage / adaptive coverage, stopped ratio,
    speed / adaptive free-flow speed  (all ratios relative to the scene's own history)
  * queue = vehicles that are stopped (direction-agnostic), clearance = queue / discharge rate

Examples
  python live_traffic.py --source highway.mp4
  python live_traffic.py --source 0                                # webcam
  python live_traffic.py --source rtsp://user:pw@ip:554/stream --tracker bytetrack
  python live_traffic.py --source a.mp4 --no-show --log logs/a.csv # make training data
  python live_traffic.py --source a.mp4 --model trafficnet.pt      # use trained forecaster
"""
import argparse
import csv
import math
import os
import sys
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

from trafficnet import FEATURES, HORIZON, MAX_CLEAR, WINDOW

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# typical real-world bbox-diagonal (m) per class: turns body-lengths/s into an approximate km/h
REF_DIAG_M = {"car": 5.2, "motorcycle": 2.3, "bus": 12.5, "truck": 9.0}
LEVELS = [(30, "Low", (60, 200, 60)), (55, "Medium", (0, 215, 255)),
          (75, "High", (0, 140, 255)), (101, "Severe", (0, 0, 255))]

STOP_BL, MOVE_BL = 0.10, 0.18      # body-lengths/s hysteresis for "stopped" / "moving"
MIN_AGE_COUNT = 0.8                # seconds a track must live before it is counted
CONGESTED, CLEARED = 50.0, 35.0    # density-score thresholds used for clearance


def level_of(score):
    for hi, name, col in LEVELS:
        if score < hi:
            return name, col
    return LEVELS[-1][1], LEVELS[-1][2]


# ============================================================ input (threaded, latest-frame)
class FrameSource:
    """Live sources: a reader thread always keeps only the newest frame (no growing lag).
    Files: read sequentially, timestamps come from the video clock."""

    def __init__(self, src, realtime=False):
        s = str(src)
        self.live = s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.cap = cv2.VideoCapture(int(s) if s.isdigit() else s)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open source: {src}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.realtime = realtime and not self.live
        self.idx, self.eof, self.stopped = 0, False, False
        self.t_start = time.time()
        if self.live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.lock, self.frame, self.fid, self.ts, self.last = threading.Lock(), None, 0, 0.0, 0
            threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        fails = 0
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    self.eof = True
                    return
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame, self.fid, self.ts = f, self.fid + 1, time.time() - self.t_start

    def read(self):
        if self.live:
            while not self.eof:
                with self.lock:
                    if self.fid > self.last and self.frame is not None:
                        self.last = self.fid
                        return self.frame, self.ts
                time.sleep(0.001)
            return None
        if self.realtime:                                   # pace a file like a live camera
            due = self.t_start + self.idx / self.fps
            time.sleep(max(0.0, due - time.time()))
        ok, f = self.cap.read()
        if not ok:
            return None
        ts = self.idx / self.fps
        self.idx += 1
        return f, ts

    def close(self):
        self.stopped = True
        self.cap.release()


# ============================================================ detection + tracking
class DetectorTracker:
    def __init__(self, a, device):
        from ultralytics import YOLO
        self.model = YOLO(a.weights, task="detect")
        self.a, self.device = a, device
        self.half = (device != "cpu") and not a.no_half
        self.classes = list(VEHICLE_CLASSES)
        self.ds = None
        if a.tracker == "deepsort":
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.ds = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7,
                               embedder="mobilenet", half=self.half,
                               embedder_gpu=(device != "cpu"))

    def __call__(self, frame):
        """returns [(track_id, class_name, (x1,y1,x2,y2))]"""
        a = self.a
        kw = dict(imgsz=a.imgsz, conf=a.conf, iou=a.iou, classes=self.classes,
                  device=self.device, half=self.half, verbose=False)
        out = []
        if self.ds is None:                                  # ByteTrack built into Ultralytics
            r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
            b = r.boxes
            if b is None or b.id is None:
                return out
            for xyxy, tid, c in zip(b.xyxy.cpu().numpy(), b.id.cpu().numpy().astype(int),
                                    b.cls.cpu().numpy().astype(int)):
                out.append((int(tid), VEHICLE_CLASSES.get(int(c), "car"), tuple(map(float, xyxy))))
            return out
        r = self.model.predict(frame, **kw)[0]               # YOLO -> DeepSORT
        b = r.boxes
        raw = []
        if b is not None and len(b):
            for xyxy, cf, c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                   b.cls.cpu().numpy().astype(int)):
                x1, y1, x2, y2 = map(float, xyxy)
                raw.append(([x1, y1, x2 - x1, y2 - y1], float(cf), int(c)))
        for t in self.ds.update_tracks(raw, frame=frame):
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            c = t.get_det_class() if hasattr(t, "get_det_class") else None
            l, tp, rr, bt = t.to_ltrb()
            out.append((int(t.track_id), VEHICLE_CLASSES.get(c, "car"), (l, tp, rr, bt)))
        return out


# ============================================================ per-vehicle state
class TrackState:
    def __init__(self, tid, cls, t):
        self.tid, self.cls = tid, cls
        self.first_t = self.last_t = t
        self.hist = deque(maxlen=90)           # (t, cx, cy, diag)
        self.sm = None
        self.trail = deque(maxlen=25)
        self.speed_bl = 0.0
        self.valid = False
        self.stopped, self.stopped_since = False, None
        self.event = None
        self.counted = False
        self.box = None

    def update(self, t, box, cls):
        x1, y1, x2, y2 = box
        self.box, self.cls, self.last_t = box, cls, t
        cx, cy, diag = (x1 + x2) / 2, (y1 + y2) / 2, math.hypot(x2 - x1, y2 - y1)
        self.sm = (cx, cy) if self.sm is None else (0.5 * cx + 0.5 * self.sm[0], 0.5 * cy + 0.5 * self.sm[1])
        self.hist.append((t, self.sm[0], self.sm[1], diag))
        self.trail.append((int(self.sm[0]), int(self.sm[1])))
        # displacement over the last ~1 s, expressed in body-lengths (diag) per second
        w = [h for h in self.hist if t - h[0] <= 1.0]
        if len(w) < 2 or t - w[0][0] < 0.4:
            return
        dt = t - w[0][0]
        dref = float(np.mean([h[3] for h in w])) + 1e-6
        disp = max(0.0, math.hypot(w[-1][1] - w[0][1], w[-1][2] - w[0][2]) - 0.03 * dref)  # jitter deadband
        raw = disp / dref / dt
        self.speed_bl = raw if not self.valid else 0.6 * self.speed_bl + 0.4 * raw
        self.valid = True
        age = t - self.first_t
        if self.stopped:
            if self.speed_bl > MOVE_BL:
                self.stopped = False
                if t - self.stopped_since >= 1.0:
                    self.event = "depart"
        elif self.speed_bl < STOP_BL and age >= 1.0:
            self.stopped, self.stopped_since = True, t

    @property
    def kmh(self):
        return self.speed_bl * REF_DIAG_M.get(self.cls, 5.2) * 3.6


# ============================================================ scene analytics (scale-free)
class SceneAnalytics:
    def __init__(self, a):
        self.a = a
        self.tracks, self.unique = {}, set()
        self.arrivals, self.departures = deque(), deque()
        self.hist = deque(maxlen=600)                 # 1 Hz samples, ~10 min of baseline memory
        self.score_hist = deque(maxlen=120)
        self.sr_hist = deque(maxlen=30)               # (mean_bl, stopped_ratio) at 1 Hz
        self.cap, self.occ_cap, self.ff_bl = max(a.capacity or 0, 10), 0.05, 1.0
        self.score, self.t_prev = 0.0, None
        self.calibrating = True
        self.inc_until, self.inc_reason = -1.0, ""
        self.m = {}

    # ---- every frame
    def update(self, t, dets, shape):
        H, W = shape[:2]
        seen, area = [], 0.0
        for tid, cls, box in dets:
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackState(tid, cls, t)
            st.update(t, box, cls)
            seen.append(st)
            area += max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if not st.counted and t - st.first_t >= MIN_AGE_COUNT:
                st.counted = True
                self.unique.add(tid)
                self.arrivals.append(t)
            if st.event == "depart":
                self.departures.append(t)
                st.event = None
        for tid in [k for k, s in self.tracks.items() if t - s.last_t > 2.0]:
            del self.tracks[tid]
        for dq in (self.arrivals, self.departures):
            while dq and t - dq[0] > 30.0:
                dq.popleft()

        valid = [s for s in seen if s.valid]
        n = len(seen)
        mean_bl = float(np.mean([s.speed_bl for s in valid])) if valid else 0.0
        mean_kmh = float(np.mean([s.kmh for s in valid])) if valid else 0.0
        stopped = sum(s.stopped for s in valid)
        stopped_ratio = stopped / len(valid) if valid else 0.0
        occ = min(1.0, area / float(W * H))

        crowd = min(1.0, n / self.cap)
        occ_rel = min(1.0, occ / self.occ_cap)
        speed_rel = min(1.0, mean_bl / self.ff_bl) if valid else 1.0
        # crowding (how full) + stopped/slow share (how jammed); the latter carries most weight so a
        # busy-but-flowing road stays Low/Medium while a same-size stopped road becomes High/Severe
        raw_score = 100 * float(np.clip(0.30 * crowd + 0.15 * occ_rel + 0.30 * stopped_ratio
                                        + 0.25 * (1 - speed_rel), 0, 1))
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        alpha = 1 - math.exp(-dt / 2.0) if dt else 1.0     # ~2 s smoothing, frame-rate independent
        self.score = raw_score if self.t_prev is None else self.score + alpha * (raw_score - self.score)
        self.t_prev = t

        span = max(1.0, min(30.0, t))
        self.m = dict(
            t=t, seen=seen, count=n, by_class=Counter(s.cls for s in seen), unique=len(self.unique),
            flow_per_min=len(self.arrivals) * 60.0 / span, mean_bl=mean_bl, mean_kmh=mean_kmh,
            queue=stopped, stopped_ratio=stopped_ratio, occ=occ, crowd=crowd, occ_rel=occ_rel,
            speed_rel=speed_rel, score=self.score, level=level_of(self.score)[0],
            depart_rel=min(1.0, len(self.departures) / self.cap),
            arrive_rel=min(1.0, len(self.arrivals) / self.cap),
            depart_rate=len(self.departures) / span, arrive_rate=len(self.arrivals) / span,
            calibrating=self.calibrating, incident=t < self.inc_until, inc_reason=self.inc_reason)
        return self.m

    # ---- once per second: update baselines, incident rule, produce the model feature vector
    def tick(self, t):
        m = self.m
        if not m:
            return None
        self.hist.append((m["count"], m["occ"], m["mean_bl"] if m["queue"] + m["count"] > 0 else 0.0))
        if len(self.hist) >= 10:
            counts = np.array([h[0] for h in self.hist])
            occs = np.array([h[1] for h in self.hist])
            bls = np.array([h[2] for h in self.hist if h[2] > 0.05])
            # 2x headroom: the busiest thing this camera has seen is NOT "full", otherwise a steady
            # scene would always look saturated relative to itself. --capacity overrides this.
            if not self.a.capacity:
                self.cap = max(10.0, 2.0 * float(np.percentile(counts, 95)))
            self.occ_cap = max(0.04, 2.0 * float(np.percentile(occs, 95)))
            if self.a.free_flow_kmh:
                self.ff_bl = self.a.free_flow_kmh / 3.6 / REF_DIAG_M["car"]
            elif len(bls) >= 5:
                self.ff_bl = max(0.3, float(np.percentile(bls, 90)))
        elif not self.a.capacity:
            self.cap = max(10.0, 2.0 * max(h[0] for h in self.hist))
        self.calibrating = len(self.hist) < 20

        # adaptive incident rule: sudden slowdown, sudden stop-wave, or a lone stalled vehicle
        inc, reason = 0, ""
        prev = list(self.sr_hist)[-12:]
        if len(prev) >= 8 and m["count"] >= 4 and not self.calibrating:
            med_bl = float(np.median([p[0] for p in prev]))
            med_sr = float(np.median([p[1] for p in prev]))
            if med_bl > 0.25 and m["mean_bl"] < 0.5 * med_bl:
                inc, reason = 1, "sudden speed drop"
            elif m["stopped_ratio"] - med_sr > 0.3:
                inc, reason = 1, "stop wave / queue spike"
            elif (m["stopped_ratio"] < 0.25 and m["speed_rel"] > 0.5 and
                  any(s.stopped and t - s.stopped_since >= 4.0 for s in m["seen"])):
                inc, reason = 1, "stalled vehicle"
        self.sr_hist.append((m["mean_bl"], m["stopped_ratio"]))
        if inc:
            self.inc_until, self.inc_reason = t + 5.0, reason
        self.score_hist.append(m["score"])
        m["incident_rule"] = inc
        return np.array([m["score"] / 100.0, m["crowd"], m["occ_rel"], m["stopped_ratio"],
                         m["speed_rel"], m["depart_rel"], m["arrive_rel"], inc], dtype=np.float32)


# ============================================================ forecaster (TrafficNet or heuristic)
class Forecaster:
    def __init__(self, path, device):
        self.net, self.buf, self.device = None, deque(maxlen=WINDOW), device
        self.source = "heuristic"
        if path and os.path.exists(path):
            import torch
            from Het_Patel.trafficnet import build_model
            ck = torch.load(path, map_location="cpu")
            self.net = build_model()
            self.net.load_state_dict(ck["state"])
            self.dev = "cuda" if (torch.cuda.is_available() and device != "cpu") else "cpu"
            self.net.to(self.dev).eval()
            self.torch, self.source = torch, "TrafficNet"
        elif path:
            print(f"[warn] model file {path} not found - using heuristic forecaster", file=sys.stderr)

    def predict(self, feat, m, score_hist):
        self.buf.append(feat)
        # --- heuristic (also the warm-up fallback until 30 s of history exist)
        s = list(score_hist)[-15:]
        slope = float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) >= 6 else 0.0   # pts / s
        f_score = float(np.clip(m["score"] + 0.5 * slope * HORIZON, 0, 100))
        cands = []
        if m["score"] >= CONGESTED:
            if m["depart_rate"] > 0.02:
                cands.append(m["queue"] / m["depart_rate"])
            if slope < -0.05:
                cands.append((m["score"] - CLEARED) / -slope)
            clear = float(min(MAX_CLEAR, np.mean(cands))) if cands else None    # None = not clearing
        else:
            clear = 0.0
        inc_p = float(m.get("incident_rule", 0))
        src = "heuristic"
        # --- learned model
        if self.net is not None and len(self.buf) == WINDOW:
            x = self.torch.tensor(np.stack(self.buf), dtype=self.torch.float32).unsqueeze(0).to(self.dev)
            with self.torch.no_grad():
                ps, pc, pi = self.net(x)
            f_score = float(np.clip(ps.item() * 100, 0, 100))
            if m["score"] >= CONGESTED:
                clear = float(np.clip(pc.item(), 0, 1) * MAX_CLEAR)
                if clear >= MAX_CLEAR - 1:
                    clear = None
            else:
                clear = 0.0
            inc_p = float(self.torch.sigmoid(pi).item())
            src = self.source
        return dict(score=f_score, level=level_of(f_score)[0], clear=clear, inc_p=inc_p, src=src)


# ============================================================ drawing
def draw(frame, m, fc, perf, args):
    H, W = frame.shape[:2]
    sc = max(0.5, W / 1700.0)
    th = max(1, int(round(sc * 2)))
    for s in m["seen"]:
        if s.box is None:
            continue
        x1, y1, x2, y2 = map(int, s.box)
        col = (0, 0, 255) if s.stopped else (60, 220, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, th)
        if len(s.trail) > 1:
            cv2.polylines(frame, [np.array(s.trail, np.int32)], False, col, th)
        lab = f"#{s.tid} {s.kmh:.0f}km/h" if s.valid else f"#{s.tid}"
        cv2.putText(frame, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc + 0.15, col, 1, cv2.LINE_AA)

    lvl, lcol = level_of(m["score"])
    bc = m["by_class"]
    fl, fcol = (level_of(fc["score"]) if fc else ("-", (200, 200, 200)))
    if fc is None:
        clr = "..."
    elif fc["clear"] is None:
        clr = "not clearing (growing / stalled)"
    elif fc["clear"] < 1:
        clr = "clear"
    else:
        clr = f"~{fc['clear']:.0f} s"
    lines = [
        (f"Vehicles now: {m['count']}   (car {bc['car']} | moto {bc['motorcycle']} | bus {bc['bus']} | truck {bc['truck']})", (255, 255, 255)),
        (f"Total seen: {m['unique']}    Flow: {m['flow_per_min']:.0f} veh/min", (255, 255, 255)),
        (f"Density: {lvl}  {m['score']:.0f}/100" + ("  (calibrating)" if m["calibrating"] else ""), lcol),
        (f"Avg speed: {m['mean_kmh']:.0f} km/h   Queue: {m['queue']} stopped", (255, 255, 255)),
        (f"Forecast +{HORIZON}s: {fl} {fc['score']:.0f}/100" if fc else "Forecast: warming up", fcol),
        (f"Clearance: {clr}", (255, 255, 0)),
        (f"INCIDENT: {m['inc_reason']}" if m["incident"] else "Incident: none", (0, 0, 255) if m["incident"] else (200, 200, 200)),
        (f"{perf['fps']:.1f} FPS | det+track {perf['det']:.0f} ms | total {perf['total']:.0f} ms", (180, 180, 180)),
    ]
    lh, pad = int(26 * sc + 6), 8
    pw = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc + 0.2, 1)[0][0] for t, _ in lines)) + 2 * pad
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (pw, lh * len(lines) + 2 * pad), (20, 20, 20), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame, txt, (pad, pad + lh * (i + 1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * sc + 0.2, col, 1, cv2.LINE_AA)
    bx, by, bw = pad, lh * len(lines) + 2 * pad + 6, pw - 2 * pad
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 10), (60, 60, 60), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * m["score"] / 100), by + 10), lcol, -1)
    if m["incident"]:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 4)


def mmss(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# ============================================================ main loop
LOG_COLS = ["t", "count", "car", "motorcycle", "bus", "truck", "unique", "flow_per_min", "avg_kmh", "queue",
            "level"] + FEATURES + ["pred_score", "pred_clear", "incident_prob"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file, camera index (0), or rtsp/http URL")
    ap.add_argument("--weights", default="yolo_model.engine")
    ap.add_argument("--tracker", choices=["deepsort", "bytetrack"], default="deepsort",
                    help="bytetrack is faster (no appearance CNN); deepsort keeps IDs through occlusion")
    ap.add_argument("--imgsz", type=int, default=640, help="use 960-1280 for high/aerial cameras")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-half", action="store_true", help="disable FP16 on GPU")
    ap.add_argument("--max-width", type=int, default=1280, help="downscale big frames before processing")
    ap.add_argument("--model", default=None, help="trained TrafficNet .pt (optional; heuristic otherwise)")
    ap.add_argument("--capacity", type=float, default=None, help="vehicles that count as 'full' (skips auto-calibration)")
    ap.add_argument("--free-flow-kmh", type=float, default=None, help="normal speed of this road (skips auto-calibration)")
    ap.add_argument("--realtime", action="store_true", help="play a file at its native speed (live-camera simulation)")
    ap.add_argument("--no-show", action="store_true", help="headless (no window)")
    ap.add_argument("--save", default=None, help="write annotated video to this path (.mp4)")
    ap.add_argument("--log", default=None, help="write 1 Hz feature log CSV (training data for TrafficNet)")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()

    if a.device == "auto":
        try:
            import torch
            device = 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = a.device
    print(f"[init] device={device} tracker={a.tracker} imgsz={a.imgsz} model={a.model or 'heuristic forecaster'}")

    src = FrameSource(a.source, a.realtime)
    det = DetectorTracker(a, device)
    ana = SceneAnalytics(a)
    fcst = Forecaster(a.model, device)
    fc, last_tick = None, -1e9
    writer, logf, logw = None, None, None
    if a.log:
        os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
        logf = open(a.log, "w", newline="")
        logw = csv.writer(logf)
        logw.writerow(LOG_COLS)

    perf = dict(fps=0.0, det=0.0, total=0.0)
    n_frames, t_loop = 0, time.perf_counter()
    if not a.no_show:
        cv2.namedWindow("Live Traffic Intelligence", cv2.WINDOW_NORMAL)
    print("time      vehicles  total  flow/min | density        | speed  queue | forecast(+%ds)   | clearance | incident | perf" % HORIZON)
    try:
        while True:
            item = src.read()
            if item is None:
                break
            frame, ts = item
            if frame.shape[1] > a.max_width:
                s = a.max_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

            t0 = time.perf_counter()
            dets = det(frame)
            t1 = time.perf_counter()
            m = ana.update(ts, dets, frame.shape)

            if ts - last_tick >= 1.0:                                # 1 Hz: features, forecast, terminal, log
                last_tick = ts
                feat = ana.tick(ts)
                fc = fcst.predict(feat, m, ana.score_hist)
                clr = "n/a" if m["score"] < CONGESTED and not fc["clear"] else (
                    "never?" if fc["clear"] is None else f"{fc['clear']:.0f}s")
                bc = m["by_class"]
                print(f"{mmss(ts)}  {m['count']:>4} (c{bc['car']} m{bc['motorcycle']} b{bc['bus']} t{bc['truck']})  "
                      f"{m['unique']:>5}  {m['flow_per_min']:>6.0f} | {m['level']:<7}{m['score']:>5.1f} | "
                      f"{m['mean_kmh']:>4.0f}   {m['queue']:>4} | {fc['level']:<7}{fc['score']:>5.1f} [{fc['src'][:4]}] | "
                      f"{clr:>8}  | {('YES: ' + m['inc_reason']) if m['incident'] else 'no':<8} | "
                      f"{perf['fps']:.1f}fps {perf['total']:.0f}ms", flush=True)
                if logw:
                    logw.writerow([f"{ts:.2f}", m["count"], bc["car"], bc["motorcycle"], bc["bus"], bc["truck"],
                                   m["unique"], f"{m['flow_per_min']:.1f}", f"{m['mean_kmh']:.1f}", m["queue"], m["level"],
                                   f"{m['score']:.2f}", f"{m['crowd']:.4f}", f"{m['occ_rel']:.4f}", f"{m['stopped_ratio']:.4f}",
                                   f"{m['speed_rel']:.4f}", f"{m['depart_rel']:.4f}", f"{m['arrive_rel']:.4f}",
                                   m.get("incident_rule", 0), f"{fc['score']:.2f}",
                                   "" if fc["clear"] is None else f"{fc['clear']:.1f}", f"{fc['inc_p']:.3f}"])

            t2 = time.perf_counter()
            n_frames += 1
            perf["det"] = 0.9 * perf["det"] + 0.1 * (t1 - t0) * 1000 if n_frames > 1 else (t1 - t0) * 1000
            perf["total"] = 0.9 * perf["total"] + 0.1 * (t2 - t0) * 1000 if n_frames > 1 else (t2 - t0) * 1000
            now = time.perf_counter()
            inst = 1.0 / max(now - t_loop, 1e-6)
            perf["fps"] = inst if n_frames == 1 else 0.9 * perf["fps"] + 0.1 * inst
            t_loop = now

            if not a.no_show or a.save:
                draw(frame, m, fc, perf, a)
            if a.save:
                if writer is None:
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             min(src.fps, 30), (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if not a.no_show:
                cv2.imshow("Live Traffic Intelligence", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if a.max_frames and n_frames >= a.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if writer:
            writer.release()
        if logf:
            logf.close()
        cv2.destroyAllWindows()
    print(f"\n[done] {n_frames} frames | unique vehicles: {len(ana.unique)} | avg {perf['fps']:.1f} FPS")


if __name__ == "__main__":
    main()
