import argparse
import base64
import io
import json
import os
import statistics
import subprocess
import threading
import time
import warnings
from pathlib import Path
from PIL import Image
import requests
import numpy as np

from utils import replace_system_prompt, save_json

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pynvml = None
    NVML_AVAILABLE = False


def percentile(values, pct):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.percentile(vals, pct)) if vals else None


def safe_mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(statistics.mean(vals)) if vals else None


def safe_max(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(max(vals)) if vals else None


class SystemTelemetry:
    """Samples device-wide GPU metrics and the current process tree."""

    def __init__(self, gpu_index: int = 0, interval_sec: float = 0.25):
        self.gpu_index = gpu_index
        self.interval_sec = interval_sec
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None

        if NVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as exc:
                warnings.warn(f"NVML GPU {gpu_index} unavailable: {exc}")

    def _process_tree_stats(self):
        if psutil is None:
            return {}
        own_procs = {}
        server_procs = {}
        try:
            root = psutil.Process(os.getpid())
            for proc in [root] + root.children(recursive=True):
                own_procs[proc.pid] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        try:
            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    cmdline = " ".join(proc.info.get("cmdline") or []).lower()
                    if "llama-server" in name or "llama-server" in cmdline:
                        server_procs[proc.pid] = proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass

        def aggregate(procs):
            cpu_pct, rss_bytes = [], []
            for proc in procs.values():
                try:
                    cpu_pct.append(proc.cpu_percent(interval=None))
                    rss_bytes.append(proc.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return (
                sum(cpu_pct) if cpu_pct else None,
                sum(rss_bytes) / (1024 ** 2) if rss_bytes else None,
            )

        own_cpu, own_rss = aggregate(own_procs)
        server_cpu, server_rss = aggregate(server_procs)
        return {
            "process_tree_cpu_percent_sum": own_cpu,
            "process_tree_rss_mb_sum": own_rss,
            "llama_server_cpu_percent_sum": server_cpu,
            "llama_server_rss_mb_sum": server_rss,
        }

    def _sample_once(self):
        now = time.monotonic()
        sample = {"t_sec": now}
        if self._handle is not None:
            try:
                sample["gpu_power_w"] = (
                    pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                )
            except Exception:
                sample["gpu_power_w"] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                sample["gpu_memory_used_mb"] = mem.used / (1024 ** 2)
                sample["gpu_memory_total_mb"] = mem.total / (1024 ** 2)
            except Exception:
                pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                sample["gpu_util_percent"] = float(util.gpu)
                sample["gpu_memory_util_percent"] = float(util.memory)
            except Exception:
                pass
            try:
                sample["gpu_temperature_c"] = float(
                    pynvml.nvmlDeviceGetTemperature(
                        self._handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                )
            except Exception:
                pass
            try:
                sample["gpu_sm_clock_mhz"] = float(
                    pynvml.nvmlDeviceGetClockInfo(
                        self._handle, pynvml.NVML_CLOCK_SM
                    )
                )
            except Exception:
                pass
        sample.update(self._process_tree_stats())
        self.samples.append(sample)

    def _run(self):
        self._sample_once()
        while not self._stop.wait(self.interval_sec):
            self._sample_once()
        self._sample_once()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def summarize(self):
        if not self.samples:
            return {}
        result = {
            "telemetry_sample_count": len(self.samples),
            "telemetry_interval_sec": self.interval_sec,
        }
        metric_keys = [
            "gpu_power_w", "gpu_memory_used_mb", "gpu_util_percent",
            "gpu_memory_util_percent", "gpu_temperature_c", "gpu_sm_clock_mhz",
            "process_tree_cpu_percent_sum", "process_tree_rss_mb_sum",
        ]
        for key in metric_keys:
            vals = [s.get(key) for s in self.samples if s.get(key) is not None]
            if vals:
                result[f"{key}_mean"] = safe_mean(vals)
                result[f"{key}_max"] = safe_max(vals)
        memory_vals = [
            s["gpu_memory_used_mb"] for s in self.samples
            if s.get("gpu_memory_used_mb") is not None
        ]
        if memory_vals:
            result["gpu_memory_peak_mb_device_wide"] = max(memory_vals)

        pairs = [
            (s["t_sec"], s.get("gpu_power_w"))
            for s in self.samples if s.get("gpu_power_w") is not None
        ]
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(pairs, pairs[1:]):
            if t1 > t0:
                energy_j += (p0 + p1) * 0.5 * (t1 - t0)
        if len(pairs) >= 2:
            result["gpu_energy_j_estimated"] = energy_j
            result["gpu_energy_method"] = "trapezoidal integration of NVML sampled board power"
        else:
            result["gpu_energy_j_estimated"] = None
            result["gpu_energy_method"] = "unavailable; NVML power readings were not available"
        result["gpu_telemetry_available"] = self._handle is not None
        result["psutil_available"] = psutil is not None
        return result

def start_llama_server(model_path: str, mmproj_path: str, host: str = "127.0.0.1", port: int = 8080):
    cmd = [
        "llama", "serve",
        "--model", model_path,
        "--mmproj", mmproj_path,
        "--host", host,
        "--port", str(port),
        "--ctx-size", "2048",
        "--n-gpu-layers", "20",
        "--parallel", "1"
    ]
    
    print(f"Starting server with command:\n" + " ".join(cmd))
    
    # Capture stderr so we can inspect error logs if it fails
    server_proc = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.PIPE,
        text=True
    )

    health_url = f"http://{host}:{port}/health"
    print("Waiting for llama-server to spin up...")
    
    start_time = time.time()
    max_wait = 120
    while time.time() - start_time < max_wait:
        if server_proc.poll() is not None:
            # Read any error output from stderr
            err_output = server_proc.stderr.read() if server_proc.stderr else "No stderr captured"
            print(f"error code: {server_proc.returncode}")
            print(f"error details:\n{err_output}")
            raise RuntimeError("llama-server process exited unexpectedly during startup.")
        
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                print("llama-server is ready!")
                return server_proc
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
        
    server_proc.terminate()
    if server_proc.stderr:
        print(server_proc.stderr.read())
    raise TimeoutError("llama-server failed to become ready within the timeout period.")

def main():
    parser = argparse.ArgumentParser(description="Run DriveBench Evaluation with llama-server")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--mmproj", required=True, help="Path to mmproj file")
    parser.add_argument("--output-prefix", required=True, help="Prefix path for output results")
    args = parser.parse_args()

    data_path = "/home/db2003/Desktop/Amr/(MLsys)_DriveBench/data/data/drivebench-test-simple-perception.json"
    prompt_path = "/home/db2003/Desktop/Amr/(MLsys)_DriveBench/prompt.txt"
    
    api_base = "http://127.0.0.1:8080/v1"
    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt_template = f.read()

    # Launch server automatically using CLI arguments
    server_proc = start_llama_server(args.model, args.mmproj, host="127.0.0.1", port=8080)

    try:
        telemetry = SystemTelemetry(gpu_index=0, interval_sec=0.25)
        telemetry.start()
        run_start = time.perf_counter()

        results = []
        preprocess_latencies = []
        generation_latencies = []
        end_to_end_latencies = []
        generated_tokens = []
        input_tokens = []

        for idx, item in enumerate(dataset):
            question = item["question"]
            sample_filenames = item["image_path"]

            prep_start = time.perf_counter()
            original_paths = list(sample_filenames.values())
            image_paths = original_paths[:4]

            system_prompt = replace_system_prompt(system_prompt_template, image_paths)

            user_content = []
            for filename in image_paths:
                img_path = filename.replace("nuscenes/samples", "data/corruption/H256ABRCompression")
                try:
                    with Image.open(img_path) as source:
                        img = source.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    b64_img = base64.b64encode(buffer.getvalue()).decode("ascii")
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

            user_content.append({"type": "text", "text": question})
            preprocess_sec = time.perf_counter() - prep_start
            preprocess_latencies.append(preprocess_sec)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            gen_start = time.perf_counter()
            res = requests.post(f"{api_base}/chat/completions", json={
                "model": "local-model",
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.2,
                "max_tokens": 512,
                "stream": False
            })
            generate_sec = time.perf_counter() - gen_start
            generation_latencies.append(generate_sec)

            end_to_end_sec = preprocess_sec + generate_sec
            end_to_end_latencies.append(end_to_end_sec)

            resp_json = res.json()
            choices = resp_json.get("choices", [])
            pred_text = choices[0].get("message", {}).get("content", "") if choices else ""
            
            usage = resp_json.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            prompt_toks = usage.get("prompt_tokens", 0)
            
            generated_tokens.append(comp_tokens)
            input_tokens.append(prompt_toks)

            item["pred"] = pred_text
            item["eff_preprocess_sec"] = preprocess_sec
            item["eff_generate_sec"] = generate_sec
            item["eff_end_to_end_sec"] = end_to_end_sec
            item["eff_generated_tokens"] = comp_tokens
            item["eff_input_tokens"] = prompt_toks
            results.append(item)

            print(f"Processed item {idx+1}/{len(dataset)} in {end_to_end_sec:.2f}s (gen: {generate_sec:.2f}s)")

        run_wall_sec = time.perf_counter() - run_start
        telemetry.stop()
        telemetry_summary = telemetry.summarize()

        save_json(results, args.output_prefix + ".json")

        successful = sum(1 for r in results if r.get("pred") is not None)
        failures = len(results) - successful

        summary = {
            "model": args.model,
            "data": data_path,
            "corruption": "h256",
            "num_examples": len(results),
            "successful_examples": successful,
            "failed_examples": failures,
            "failure_rate": failures / len(results) if results else None,
            "run_wall_sec": run_wall_sec,
            "run_throughput_examples_per_sec": (
                len(results) / run_wall_sec if run_wall_sec > 0 else None
            ),
            "per_example_end_to_end_latency_sec_mean": safe_mean(end_to_end_latencies),
            "per_example_end_to_end_latency_sec_p50": percentile(end_to_end_latencies, 50),
            "per_example_end_to_end_latency_sec_p95": percentile(end_to_end_latencies, 95),
            "per_example_generation_latency_sec_mean": safe_mean(generation_latencies),
            "per_example_generation_latency_sec_p50": percentile(generation_latencies, 50),
            "per_example_generation_latency_sec_p95": percentile(generation_latencies, 95),
            "per_example_preprocess_latency_sec_mean": safe_mean(preprocess_latencies),
            "generated_tokens_total": sum(generated_tokens) if generated_tokens else 0,
            "generated_tokens_per_sec": (
                sum(generated_tokens) / run_wall_sec if run_wall_sec > 0 else None
            ),
            "backend": "llama.cpp server",
            "llama_cpp_api_base": api_base,
            "llama_cpp_model_path_metadata": args.model,
            "precision_or_quantization": "GGUF",
            "max_model_len": 4096,
            "max_tokens": 512,
            "image_size": 224,
            "num_images_per_prompt": 4,
            "temperature": 0.2,
            "top_p": 0.2,
            "telemetry": telemetry_summary,
        }

        energy_j = telemetry_summary.get("gpu_energy_j_estimated")
        summary["gpu_energy_j_per_example_estimated"] = (
            energy_j / len(results) if energy_j is not None and results else None
        )
        summary["gpu_energy_j_per_successful_example_estimated"] = (
            energy_j / successful if energy_j is not None and successful else None
        )

        summary_path = args.output_prefix + ".efficiency.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nPredictions: {args.output_prefix}.json")
        print(f"Efficiency summary: {summary_path}")
        print(json.dumps(summary, indent=2, default=str))

    finally:
        print("\nShutting down llama-server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("llama-server stopped.")


if __name__ == "__main__":
    main()