"""
benchmarker.py
==============
Week 4 Performance Evaluation Module
--------------------------------------
Measures and logs:
  • FPS (frames per second)
  • Per-frame latency (ms)
  • GPU memory usage (MB)  – CUDA only
  • System RAM usage (MB)
  • Traversability statistics per frame

All results are saved to  seg_benchmark.json  in the project directory
for inclusion in the IEEE paper benchmarking section.

Usage
-----
    from benchmarker import Benchmarker
    bench = Benchmarker()

    bench.start_frame()
    # ... run inference ...
    bench.end_frame(score_map_stats_dict)

    bench.save("seg_benchmark.json")
    bench.print_summary()
"""

import time
import json
import os
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class Benchmarker:
    """Real-time performance logger for the segmentation pipeline."""

    def __init__(self, device_name: str = "auto"):
        self.device_name = device_name

        self._frame_start_time: float | None = None

        # Per-frame records
        self.latencies_ms: list[float] = []
        self.gpu_mem_mb:   list[float] = []
        self.ram_mb:       list[float] = []
        self.safe_pct:     list[float] = []
        self.obs_pct:      list[float] = []
        self.mean_scores:  list[float] = []

        self.frame_count: int = 0
        self.start_wall:  float = time.time()

        # Detect GPU
        self._has_cuda = (
            _TORCH_AVAILABLE and
            hasattr(torch, "cuda") and
            torch.cuda.is_available()
        )

    # ─────────────────────────────────────────────
    #  Per-frame API
    # ─────────────────────────────────────────────

    def start_frame(self) -> None:
        """Call immediately before running model inference."""
        self._frame_start_time = time.perf_counter()

    def end_frame(self, score_stats: dict | None = None) -> dict:
        """
        Call immediately after all per-frame processing.

        Parameters
        ----------
        score_stats : dict  (optional)
            Output of traversability_scorer.score_map_stats()
            Keys: mean_score, safe_pixel_pct, obstacle_pixel_pct

        Returns
        -------
        metrics : dict  – metrics for this frame
        """
        now = time.perf_counter()
        latency_ms = (now - self._frame_start_time) * 1000.0

        self.latencies_ms.append(latency_ms)
        self.frame_count += 1

        # GPU memory
        gpu_mb = 0.0
        if self._has_cuda:
            try:
                gpu_mb = torch.cuda.memory_allocated() / 1e6
            except Exception:
                pass
        self.gpu_mem_mb.append(gpu_mb)

        # System RAM
        ram_mb = 0.0
        if _PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(os.getpid())
                ram_mb = proc.memory_info().rss / 1e6
            except Exception:
                pass
        self.ram_mb.append(ram_mb)

        # Traversability stats
        if score_stats:
            self.mean_scores.append(score_stats.get("mean_score", 0.0))
            self.safe_pct.append(score_stats.get("safe_pixel_pct", 0.0))
            self.obs_pct.append(score_stats.get("obstacle_pixel_pct", 0.0))

        return {
            "frame":      self.frame_count,
            "latency_ms": round(latency_ms, 2),
            "gpu_mem_mb": round(gpu_mb, 1),
            "ram_mb":     round(ram_mb, 1),
        }

    # ─────────────────────────────────────────────
    #  Summary
    # ─────────────────────────────────────────────

    def summary(self) -> dict:
        """Compute and return the full benchmark summary dictionary."""
        elapsed = time.time() - self.start_wall
        fps_overall = self.frame_count / elapsed if elapsed > 0 else 0.0

        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])

        result = {
            "model":          "SegFormer-B0 · ADE20K-512",
            "device":         self.device_name,
            "total_frames":   self.frame_count,
            "total_time_s":   round(elapsed, 2),
            "fps": {
                "mean":       round(fps_overall, 2),
                "max":        round(1000.0 / float(np.min(lat)) if np.min(lat) > 0 else 0, 2),
            },
            "latency_ms": {
                "mean":       round(float(np.mean(lat)), 2),
                "std":        round(float(np.std(lat)),  2),
                "min":        round(float(np.min(lat)),  2),
                "max":        round(float(np.max(lat)),  2),
                "p95":        round(float(np.percentile(lat, 95)), 2),
            },
            "gpu_memory_mb": {
                "mean":       round(float(np.mean(self.gpu_mem_mb)), 1) if self.gpu_mem_mb else 0,
                "max":        round(float(np.max(self.gpu_mem_mb)), 1) if self.gpu_mem_mb else 0,
            },
            "ram_mb": {
                "mean":       round(float(np.mean(self.ram_mb)), 1) if self.ram_mb else 0,
                "max":        round(float(np.max(self.ram_mb)), 1) if self.ram_mb else 0,
            },
        }

        if self.mean_scores:
            result["traversability"] = {
                "mean_score":        round(float(np.mean(self.mean_scores)), 4),
                "safe_pixel_pct":    round(float(np.mean(self.safe_pct)),   2),
                "obstacle_pixel_pct":round(float(np.mean(self.obs_pct)),    2),
            }

        return result

    def save(self, path: str = "seg_benchmark.json") -> None:
        """Write the summary to a JSON file."""
        data = self.summary()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Benchmarker] Saved benchmark → {path}")

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        s = self.summary()
        print("\n" + "=" * 55)
        print("  PERFORMANCE BENCHMARK REPORT")
        print("=" * 55)
        print(f"  Model         : {s['model']}")
        print(f"  Device        : {s['device']}")
        print(f"  Total Frames  : {s['total_frames']}")
        print(f"  Total Time    : {s['total_time_s']} s")
        print(f"  Mean FPS      : {s['fps']['mean']}")
        print(f"  Max FPS       : {s['fps']['max']}")
        print(f"  Mean Latency  : {s['latency_ms']['mean']} ms")
        print(f"  P95  Latency  : {s['latency_ms']['p95']} ms")
        print(f"  GPU Mem (mean): {s['gpu_memory_mb']['mean']} MB")
        print(f"  RAM     (mean): {s['ram_mb']['mean']} MB")
        if "traversability" in s:
            t = s["traversability"]
            print(f"  Mean Trav Score : {t['mean_score']}")
            print(f"  Safe  Pixels    : {t['safe_pixel_pct']} %")
            print(f"  Obstacle Pixels : {t['obstacle_pixel_pct']} %")
        print("=" * 55 + "\n")
