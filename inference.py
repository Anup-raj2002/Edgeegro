import sys
import time
import cv2
import numpy as np
import openvino as ov
import openvino.properties as props
from config import MODEL_PATH, TARGET_DEVICE, CONF_THRESH

class PoseEstimator:
    def __init__(self):
        self.core = ov.Core()
        self.device = TARGET_DEVICE
        self.infer_ms = 0.0
        self.kernel = np.ones((25, 25), np.uint8)
        
        self._check_device()
        self._load_model()

    def _check_device(self):
        available = self.core.available_devices
        tokens = [t for t in self.device.replace(":", ",").split(",") if t not in ("AUTO", "MULTI", "HETERO")]
        missing = [t for t in tokens if not any(d.startswith(t) for d in available)]
        if missing:
            print(f"ERROR: device(s) {missing} not available. Available: {available}")
            sys.exit(1)

    def _load_model(self):
        model = self.core.read_model(MODEL_PATH)
        config = {props.hint.performance_mode(): props.hint.PerformanceMode.LATENCY}
        self.compiled = self.core.compile_model(model, self.device, config)
        
        self.input_layer = self.compiled.input(0)
        _, _, self.in_h, self.in_w = self.input_layer.shape
        
        self.heatmap_output = None
        for out in self.compiled.outputs:
            if out.shape[1] == 19:
                self.heatmap_output = out
        if self.heatmap_output is None:
            raise RuntimeError("Heatmap output not found")

    def get_peaks(self, frame, x0, x1):
        roi = frame[:, x0:x1]
        h, w = roi.shape[:2]
        scale = min(self.in_w / w, self.in_h / h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(roi, (nw, nh))
        padded = np.zeros((self.in_h, self.in_w, 3), np.uint8)
        padded[:nh, :nw] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)

        # Calculate Inference Time
        t0 = time.perf_counter()
        results = self.compiled([blob])
        ms = (time.perf_counter() - t0) * 1000
        self.infer_ms = ms if self.infer_ms == 0.0 else 0.9 * self.infer_ms + 0.1 * ms

        heatmaps = results[self.heatmap_output][0]
        all_peaks = []
        for i in range(18):
            hm = cv2.resize(heatmaps[i], (self.in_w, self.in_h), interpolation=cv2.INTER_CUBIC)
            dil = cv2.dilate(hm, self.kernel)
            ys, xs = np.nonzero((hm == dil) & (hm >= CONF_THRESH))
            pts = [(float(x) / scale + x0, float(y) / scale, float(hm[y, x])) for x, y in zip(xs, ys)]
            pts.sort(key=lambda p: -p[2])
            all_peaks.append(pts[:4])
            
        return all_peaks, self.infer_ms