import requests
import time
import json
import threading
import pandas as pd
import matplotlib.pyplot as plt
import re
import subprocess
import os
import signal
import psutil

# ================= Configuration Area =================
class Config:
    """
    Configuration priority: Environment Variables > Defaults
    Modify defaults below or set env vars to override.
    """
    # --- Basic Service Configuration ---
    # Path to Doris BE microbenchmark binary
    SERVER_BIN_PATH = os.getenv("SERVER_BIN_PATH", "/mnt/disk1/guanhaolin/doris/output/be/lib/file_cache_microbench")
    # BE library directory (usually parent of bin or sibling lib dir)
    DORIS_HOME_PATH = os.getenv("DORIS_HOME_PATH", "/mnt/disk1/guanhaolin/doris/output/be/lib")
    # Service listening port
    SERVER_PORT = int(os.getenv("SERVER_PORT", 10086))
    # Systemd Unit Name (used for Cgroup resource isolation)
    SERVICE_UNIT_NAME = os.getenv("SERVICE_UNIT_NAME", "doris_bench_disk_test")
    
    # --- Monitoring & Hardware Configuration ---
    # Physical disk device name to monitor (e.g., nvme0n1, sda)
    DISK_DEVICE = os.getenv("DISK_DEVICE", "nvme0n1")
    # Memory limit for read phase (Bytes), used to force disk I/O (Default 30GB)
    READ_PHASE_MEM_LIMIT = int(os.getenv("READ_PHASE_MEM_LIMIT", 30 * 1024 * 1024 * 1024))
    
    # --- Workload Configuration ---
    # Write Phase: Size per file
    WRITE_FILE_SIZE = int(os.getenv("WRITE_FILE_SIZE", 10 * 1024 * 1024)) # 10MB
    # Write Phase: Number of files
    WRITE_FILE_NUM = int(os.getenv("WRITE_FILE_NUM", 1000))
    # Read Phase: Concurrency range (Start, Stop, Step)
    READ_CONCURRENCY_RANGE = (5, 101, 10)
    
    # --- Output Configuration ---
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "disk_perf_dir")
    
    @property
    def BENCH_URL(self):
        return f"http://localhost:{self.SERVER_PORT}/MicrobenchService"

    @property
    def RESULT_FILE(self): return os.path.join(self.OUTPUT_DIR, "disk_summary_results.json")
    
    @property
    def RAW_RESULT_FILE(self): return os.path.join(self.OUTPUT_DIR, "disk_raw_job_status.json")
    
    @property
    def PLOT_FILE(self): return os.path.join(self.OUTPUT_DIR, "disk_analysis.png")

cfg = Config()

# ================= System Monitor Class =================
class SystemMonitor(threading.Thread):
    def __init__(self, device_name, interval=0.5):
        super().__init__()
        self.device_name = device_name
        self.interval = interval
        self.running = False
        self.records = []
        self._lock = threading.Lock()
        self.last_io = self._get_disk_io()
        self.last_ts = time.time()

    def _get_disk_io(self):
        try:
            disks = psutil.disk_io_counters(perdisk=True)
            return disks.get(self.device_name)
        except: return None

    def run(self):
        self.running = True
        while self.running:
            time.sleep(self.interval)
            curr_io = self._get_disk_io()
            curr_ts = time.time()
            
            if not self.last_io or not curr_io: continue
            
            delta_t = curr_ts - self.last_ts
            if delta_t <= 0: continue

            # Calculate throughput (MB/s)
            read_mb_s = (curr_io.read_bytes - self.last_io.read_bytes) / 1024 / 1024 / delta_t
            write_mb_s = (curr_io.write_bytes - self.last_io.write_bytes) / 1024 / 1024 / delta_t
            
            # Get CPU metrics
            cpu_pct = psutil.cpu_percent()
            cpu_times = psutil.cpu_times_percent()
            
            with self._lock:
                self.records.append({
                    "sys_read_mb_s": read_mb_s,
                    "sys_write_mb_s": write_mb_s,
                    "cpu_total": cpu_pct,
                    "cpu_iowait": cpu_times.iowait
                })

            self.last_io = curr_io
            self.last_ts = curr_ts

    def stop(self):
        self.running = False
        self.join()

    def get_avg_stats(self):
        with self._lock:
            if not self.records: return {}
            df = pd.DataFrame(self.records)
            return {
                "sys_read_mb_s": df['sys_read_mb_s'].mean(),
                "sys_write_mb_s": df['sys_write_mb_s'].mean(),
                "cpu_total": df['cpu_total'].mean(),
                "cpu_iowait": df['cpu_iowait'].mean()
            }

# ================= HTTP Client Class =================
class BenchClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def wait_for_server(self, retries=10):
        print("Checking server health...", end="")
        for _ in range(retries):
            try:
                requests.post(f"{self.base_url}/file_cache_clear")
                print(" OK.")
                return True
            except:
                print(".", end="", flush=True)
                time.sleep(1)
        print(" Failed!")
        return False

    def submit_job(self, payload):
        resp = requests.post(f"{self.base_url}/submit_job", json=payload)
        resp.raise_for_status()
        return resp.json()['job_id']

    def wait_job(self, job_id, timeout=10000):
        print(f"Waiting for Job {job_id}...", end="", flush=True)
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.post(f"{self.base_url}/get_job_status/{job_id}", json={"job_id": job_id})
                if resp.status_code == 200:
                    d = resp.json()
                    if d.get("status") == "COMPLETED": 
                        print(" Done.")
                        return d
                    if d.get("status") == "FAILED": raise Exception(f"Failed: {d}")
            except Exception: pass
            time.sleep(1)
        raise TimeoutError("Timeout")

    def clear_cache(self):
        try: requests.post(f"{self.base_url}/file_cache_clear?sync=true")
        except: pass

# ================= Process Management (Cgroup) =================
def start_server_in_cgroup():
    ld_lib = os.environ.get('LD_LIBRARY_PATH', '')
    env_flags = f"-E DORIS_HOME={cfg.DORIS_HOME_PATH}"
    if ld_lib: env_flags += f" -E LD_LIBRARY_PATH={ld_lib}"

    # Initially allocate 64GB to prevent OOM during write phase
    cmd = (f"sudo systemd-run --unit={cfg.SERVICE_UNIT_NAME} --scope "
           f"-p MemoryMax=64G " 
           f"{env_flags} "
           f"{cfg.SERVER_BIN_PATH} --port={cfg.SERVER_PORT}")
    
    print(f"[Process] Starting Server: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    return proc

def dynamic_limit_memory(limit_bytes):
    print(f"\n>>> LIMITING MEMORY TO {limit_bytes/1024/1024/1024:.2f} GB...")
    cgroup_path = f"/sys/fs/cgroup/memory/system.slice/{cfg.SERVICE_UNIT_NAME}.scope"
    cmd = f"echo {limit_bytes} | sudo tee {os.path.join(cgroup_path, 'memory.limit_in_bytes')}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(">>> Limit failed, dropping caches and retrying...")
        drop_system_caches()
        subprocess.run(cmd, shell=True)

def drop_system_caches():
    subprocess.run("sync", shell=True)
    subprocess.run("sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches'", shell=True)

def stop_server(proc):
    print("[Process] Stopping Server...")
    subprocess.run(f"sudo systemctl stop {cfg.SERVICE_UNIT_NAME}.scope", shell=True, stderr=subprocess.DEVNULL)
    try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except: pass
    subprocess.run(f"sudo killall -9 {os.path.basename(cfg.SERVER_BIN_PATH)}", shell=True, stderr=subprocess.DEVNULL)

# ================= Main Workflow =================
def run_benchmark():
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    client = BenchClient(cfg.BENCH_URL)
    client.clear_cache()
    srv_proc = start_server_in_cgroup()
    
    if not client.wait_for_server():
        stop_server(srv_proc)
        return

    try:
        # Phase 1: Write Data (Warmup)
        print("\n=== PHASE 1: Write Data ===")
        write_payload = {
            "size_bytes_perfile": cfg.WRITE_FILE_SIZE,
            "write_iops": 10000,
            "num_files": cfg.WRITE_FILE_NUM,
            "num_threads" : 100,
            "file_prefix": "haolin/disktest",
            "write_batch_size": 41943040
        }
        client.wait_job(client.submit_job(write_payload))

        # Phase 2: Read Test
        # Optional: Limit memory to force physical I/O
        # drop_system_caches() 
        # dynamic_limit_memory(cfg.READ_PHASE_MEM_LIMIT)

        print(f"\n=== PHASE 2: Disk Benchmark (Device: {cfg.DISK_DEVICE}) ===")
        print(f"{'Concur':<6} | {'App BW':<10} | {'Sys BW':<10} | {'Lat(ms)':<8} | {'CPU(IOWait)':<12}")
        
        results, raw_records = [], []
        
        for concurrency in range(*cfg.READ_CONCURRENCY_RANGE): 
            read_payload = {
                "read_iops": 100000, 
                "num_files": cfg.WRITE_FILE_NUM,
                "num_threads": concurrency,
                "file_prefix": "haolin/disktest",
                "read_offset": [1, cfg.WRITE_FILE_SIZE],
                "read_length": [cfg.WRITE_FILE_SIZE-1, cfg.WRITE_FILE_SIZE],
                "cache_type": "NORMAL"
            }

            monitor = SystemMonitor(cfg.DISK_DEVICE)
            monitor.start()

            try:
                job_result = client.wait_job(client.submit_job(read_payload))
            except Exception:
                monitor.stop()
                continue

            monitor.stop()
            mon_stats = monitor.get_avg_stats()

            # Data Processing
            job_result["_meta_concurrency"] = concurrency
            raw_records.append(job_result)
            
            stats = job_result.get("statistics", {})
            dur = float(re.search(r"([\d\.]+)", str(stats.get("total_read_time", 0))).group(1) or 0)
            app_bw = (stats.get("bytes_read_from_local", 0)/1048576) / dur if dur > 0 else 0
            lat_ms = (stats.get("local_io_timer", 0) / stats.get("num_local_io_total", 1)) / 1e6
            
            sys_bw = mon_stats.get("sys_read_mb_s", 0)
            cpu_iowait = mon_stats.get("cpu_iowait", 0)

            print(f"{concurrency:<6} | {app_bw:<8.2f}MB | {sys_bw:<8.2f}MB | {lat_ms:<8.3f} | {cpu_iowait:.1f}%")
            
            results.append({
                "concurrency": concurrency, 
                "app_bw": app_bw, "sys_bw": sys_bw, 
                "lat": lat_ms, "cpu_iowait": cpu_iowait
            })

        # Save & Visualize
        with open(cfg.RESULT_FILE, "w") as f: json.dump(results, f, indent=2)
        with open(cfg.RAW_RESULT_FILE, "w") as f: json.dump(raw_records, f, indent=2)
        visualize(results)

    finally:
        stop_server(srv_proc)

def visualize(results):
    if not results: return
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # BW Chart
    ax1.plot(df['concurrency'], df['app_bw'], 'b-o', label='App BW (Logical)')
    ax1.plot(df['concurrency'], df['sys_bw'], 'g--s', label='System BW (Physical)')
    ax1.set_ylabel('Throughput (MB/s)')
    ax1.set_title(f"Disk Throughput: Logical vs Physical ({cfg.DISK_DEVICE})")
    ax1.legend(); ax1.grid(True)

    # Latency Chart
    ax2.set_xlabel('Concurrency'); ax2.set_ylabel('Latency (ms)', color='tab:red')
    ax2.plot(df['concurrency'], df['lat'], 'r-x', label='Latency')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    ax2_r = ax2.twinx()
    ax2_r.set_ylabel('CPU IO Wait (%)', color='tab:orange')
    ax2_r.fill_between(df['concurrency'], df['cpu_iowait'], color='tab:orange', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.PLOT_FILE)
    print(f"\n[Output] Results saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    run_benchmark()