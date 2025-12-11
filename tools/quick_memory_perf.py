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
import shutil
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
    # BE library directory
    DORIS_HOME_PATH = os.getenv("DORIS_HOME_PATH", "/mnt/disk1/guanhaolin/doris/output/be/lib")
    # Path to be.conf (used if config modification is needed)
    CONF_FILE_PATH = os.getenv("CONF_FILE_PATH", os.path.join(DORIS_HOME_PATH, "conf/be.conf"))
    
    # Service listening port
    SERVER_PORT = int(os.getenv("SERVER_PORT", 10086))
    
    # --- Workload Configuration ---
    # Write Phase: Size per file (10MB)
    WRITE_FILE_SIZE = int(os.getenv("WRITE_FILE_SIZE", 10 * 1024 * 1024))
    # Write Phase: Number of files
    WRITE_FILE_NUM = int(os.getenv("WRITE_FILE_NUM", 1000))
    # Read Phase: Concurrency range (Start, Stop, Step)
    READ_CONCURRENCY_RANGE = (2, 21, 2)
    # Read Phase: Number of repetitions per concurrency
    READ_REPEAT = int(os.getenv("READ_REPEAT", 10))

    # --- Output Configuration ---
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "memory_perf_dir")

    @property
    def BENCH_URL(self):
        return f"http://localhost:{self.SERVER_PORT}/MicrobenchService"

    @property
    def RESULT_FILE(self): return os.path.join(self.OUTPUT_DIR, "advanced_memory_results.json")
    
    @property
    def RAW_RESULT_FILE(self): return os.path.join(self.OUTPUT_DIR, "raw_job_status.json")
    
    @property
    def PLOT_FILE(self): return os.path.join(self.OUTPUT_DIR, "advanced_memory_analysis.png")

cfg = Config()

# ================= Utility: Config Modifier =================
class ConfigModifier:
    def __init__(self, config_path):
        self.config_path = config_path
        self.backup_path = config_path + ".bak"

    def modify_to_memory(self):
        print(f"[Config] Backing up {self.config_path} -> {self.backup_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        shutil.copyfile(self.config_path, self.backup_path)

        with open(self.config_path, 'r') as f:
            content = f.read()

        # Replace "storage":"disk" with "storage":"memory"
        new_content = re.sub(
            r'("storage"\s*:\s*)"disk"', 
            r'\1"memory"', 
            content
        )

        if content == new_content:
            print("[Config] Warning: No 'storage':'disk' found, or already memory?")
        else:
            print("[Config] Modified 'storage':'disk' to 'storage':'memory'")

        with open(self.config_path, 'w') as f:
            f.write(new_content)

    def restore(self):
        if os.path.exists(self.backup_path):
            print(f"[Config] Restoring {self.backup_path} -> {self.config_path}")
            shutil.move(self.backup_path, self.config_path)

# ================= Resource Monitor (CPU & Context Switches) =================
class ResourceMonitor(threading.Thread):
    def __init__(self, pid, interval=0.1):
        super().__init__()
        self.root_pid = pid
        self.target_pid = pid 
        self.interval = interval
        self.running = False
        self.records = []
        self._lock = threading.Lock()
        
        # Try to find the actual worker process (in case pid is a shell wrapper)
        try:
            root_proc = psutil.Process(self.root_pid)
            children = root_proc.children(recursive=True)
            if children:
                # Assume the child with the highest RSS memory is the target
                target = max(children, key=lambda p: p.memory_info().rss)
                print(f"[Monitor] Auto-detected child process: {target.pid} ({target.name()})")
                self.target_pid = target.pid
            else:
                print(f"[Monitor] Monitoring root process: {self.root_pid} ({root_proc.name()})")
        except:
            pass

    def _get_threads_ctx_switches(self, pid):
        """
        Manually read /proc/[pid]/task/[tid]/status to aggregate context switches across all threads.
        """
        vol = 0
        invol = 0
        try:
            task_path = f"/proc/{pid}/task"
            if os.path.exists(task_path):
                for tid in os.listdir(task_path):
                    t_status = os.path.join(task_path, tid, "status")
                    try:
                        with open(t_status, 'r') as f:
                            for line in f:
                                if line.startswith("voluntary_ctxt_switches:"):
                                    vol += int(line.split()[1])
                                elif line.startswith("nonvoluntary_ctxt_switches:"):
                                    invol += int(line.split()[1])
                    except (FileNotFoundError, ProcessLookupError):
                        continue # Thread might have ended
        except Exception:
            pass
        return vol, invol

    def run(self):
        self.running = True
        try:
            proc = psutil.Process(self.target_pid)
            # Initial sample
            last_vol, last_invol = self._get_threads_ctx_switches(self.target_pid)
            last_times = proc.cpu_times()
            last_ts = time.time()
        except:
            return

        while self.running:
            time.sleep(self.interval)
            try:
                curr_times = proc.cpu_times()
                rss_gb = proc.memory_info().rss / (1024**3)
                
                curr_vol, curr_invol = self._get_threads_ctx_switches(self.target_pid)
                
                curr_ts = time.time()
                delta_t = curr_ts - last_ts
                if delta_t <= 0: continue

                vol_sw_rate = (curr_vol - last_vol) / delta_t
                invol_sw_rate = (curr_invol - last_invol) / delta_t
                
                if vol_sw_rate < 0: vol_sw_rate = 0
                if invol_sw_rate < 0: invol_sw_rate = 0

                user_cpu_pct = ((curr_times.user - last_times.user) / delta_t) * 100
                sys_cpu_pct = ((curr_times.system - last_times.system) / delta_t) * 100
                
                with self._lock:
                    self.records.append({
                        "timestamp": curr_ts,
                        "cpu_user": user_cpu_pct,
                        "cpu_sys": sys_cpu_pct,
                        "ctx_vol": vol_sw_rate,
                        "ctx_invol": invol_sw_rate,
                        "mem_rss_gb": rss_gb
                    })

                last_vol, last_invol = curr_vol, curr_invol
                last_times = curr_times
                last_ts = curr_ts

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    def stop(self):
        self.running = False
        self.join()

    def get_avg_stats(self):
        with self._lock:
            if not self.records: return {}
            df = pd.DataFrame(self.records)
            return {
                "cpu_user": df['cpu_user'].mean(),
                "cpu_sys": df['cpu_sys'].mean(),
                "ctx_vol": df['ctx_vol'].mean(),
                "ctx_invol": df['ctx_invol'].mean(),
                "mem_rss_gb": df['mem_rss_gb'].max()
            }

# ================= HTTP Client Class =================
class BenchClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def wait_for_server(self, retries=20):
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
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.post(f"{self.base_url}/get_job_status/{job_id}", json={"job_id": job_id})
                if resp.status_code == 200:
                    d = resp.json()
                    if d.get("status") == "COMPLETED": 
                        return d
                    if d.get("status") == "FAILED": raise Exception(f"Failed: {d}")
            except Exception: pass
            time.sleep(1)
        raise TimeoutError("Timeout")

    def clear_cache(self):
        try: requests.post(f"{self.base_url}/file_cache_clear?sync=true")
        except: pass

# ================= Process Management =================
def start_server_direct():
    env = os.environ.copy()
    env["DORIS_HOME"] = cfg.DORIS_HOME_PATH
    if os.environ.get("LD_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")

    cmd = [cfg.SERVER_BIN_PATH, f"--port={cfg.SERVER_PORT}"]
    print(f"[Process] Starting Server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, preexec_fn=os.setsid)
    return proc

def stop_server(proc):
    if not proc: return
    print("[Process] Stopping Server...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except:
        try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except: pass
    subprocess.run(f"killall -9 {os.path.basename(cfg.SERVER_BIN_PATH)}", shell=True, stderr=subprocess.DEVNULL)

# ================= Main Workflow =================
def run_benchmark():
    if not os.path.exists(cfg.OUTPUT_DIR):
        print(f"[Init] Creating output directory: {cfg.OUTPUT_DIR}")
        os.makedirs(cfg.OUTPUT_DIR)

    config_mod = ConfigModifier(cfg.CONF_FILE_PATH)
    srv_proc = None
    
    try:
        # Optional: Switch config to memory mode
        # config_mod.modify_to_memory()
        
        srv_proc = start_server_direct()
        client = BenchClient(cfg.BENCH_URL)
        
        if not client.wait_for_server():
            print("Server start failed.")
            return

        client.clear_cache()

        # 1. Warmup / Write Phase
        print("\n=== PHASE 1: Pre-heat Memory ===")
        write_payload = {
            "size_bytes_perfile": cfg.WRITE_FILE_SIZE,
            "write_iops": 1000,
            "num_files": cfg.WRITE_FILE_NUM,
            "num_threads" : 50,
            "file_prefix": "haolin/memtest",
            "write_batch_size": 4 * 1024 * 1024
        }
        try:
            print("Submitting Write Job...", end="")
            client.wait_job(client.submit_job(write_payload))
            print(" Done.")
        except Exception as e:
            print(f"Write Failed: {e}")
            return

        # 2. Benchmark Phase
        print("\n=== PHASE 2: Advanced Memory Benchmark ===")
        print(f"{'Concur':<6} | {'App BW':<10} | {'Lat(ms)':<8} | {'CPU(Usr/Sys)':<14} | {'CtxSw(Vol/Invol)':<18}")
        
        results = []
        raw_job_records = []
        
        for concurrency in range(*cfg.READ_CONCURRENCY_RANGE): 
            read_payload = {
                "read_iops": 2147483647, # Unlimited
                "num_files": 100,
                "num_threads": concurrency,
                "file_prefix": "haolin/memtest",
                "read_offset": [0, cfg.WRITE_FILE_SIZE],
                "read_length": [cfg.WRITE_FILE_SIZE - 1, cfg.WRITE_FILE_SIZE],
                "cache_type": "NORMAL",
                "repeat" : cfg.READ_REPEAT
            }

            monitor = ResourceMonitor(srv_proc.pid)
            monitor.start()

            try:
                job_result = client.wait_job(client.submit_job(read_payload))
            except Exception:
                monitor.stop()
                continue

            monitor.stop()
            mon_stats = monitor.get_avg_stats()

            # Save raw data with concurrency tag
            job_result["_meta_concurrency"] = concurrency
            raw_job_records.append(job_result)

            # Metrics parsing
            stats = job_result.get("statistics", {})
            dur = float(re.search(r"([\d\.]+)", str(stats.get("total_read_time", 0))).group(1)) if stats.get("total_read_time") else 0
            
            app_bw = (stats.get("bytes_read_from_local", 0)/1048576) / dur if dur > 0 else 0
            lat_ns = stats.get("local_io_timer", 0) / stats.get("num_local_io_total", 1)
            lat_ms = lat_ns / 1e6

            print(f"{concurrency:<6} | {app_bw:<8.2f}MB | {lat_ms:<8.3f} | "
                  f"{mon_stats.get('cpu_user',0):.0f}%/{mon_stats.get('cpu_sys',0):.0f}%     | "
                  f"{mon_stats.get('ctx_vol',0):.0f}/{mon_stats.get('ctx_invol',0):.0f}")

            results.append({
                "concurrency": concurrency, 
                "app_bw": app_bw, 
                "lat": lat_ms,
                "cpu_user": mon_stats.get('cpu_user', 0),
                "cpu_sys": mon_stats.get('cpu_sys', 0),
                "ctx_vol": mon_stats.get('ctx_vol', 0),
                "ctx_invol": mon_stats.get('ctx_invol', 0)
            })

        # Save results
        with open(cfg.RESULT_FILE, "w") as f: json.dump(results, f, indent=2)
        with open(cfg.RAW_RESULT_FILE, "w") as f: json.dump(raw_job_records, f, indent=2)
        
        print(f"\n[Save] Summary results -> {cfg.RESULT_FILE}")
        print(f"[Save] Raw job records -> {cfg.RAW_RESULT_FILE}")
        
        visualize(results)

    finally:
        stop_server(srv_proc)
        config_mod.restore()

def visualize(results):
    if not results: return
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    ax1.set_ylabel('Bandwidth (MB/s)', color='tab:blue', fontsize=12)
    ax1.plot(df['concurrency'], df['app_bw'], 'b-o', label='App Bandwidth')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1_r = ax1.twinx()
    ax1_r.set_ylabel('Latency (ms)', color='tab:red', fontsize=12)
    ax1_r.plot(df['concurrency'], df['lat'], 'r--x', label='Latency')
    ax1_r.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_title("1. Throughput & Latency Analysis", fontsize=14)

    ax2.set_ylabel('CPU Usage (%)', fontsize=12)
    ax2.stackplot(df['concurrency'], df['cpu_sys'], df['cpu_user'], 
                  labels=['System (Kernel/Lock)', 'User (App/Spin)'], 
                  colors=['tab:orange', 'tab:green'], alpha=0.7)
    ax2.plot(df['concurrency'], df['cpu_sys'] + df['cpu_user'], 'k--', label='Total CPU')
    ax2.legend(loc='upper left')
    ax2.set_title("2. CPU Composition: System vs User", fontsize=14)
    ax2.grid(True)

    ax3.set_ylabel('Switches / sec', fontsize=12)
    ax3.plot(df['concurrency'], df['ctx_vol'], color='purple', marker='^', label='Voluntary (Wait Lock)')
    ax3.plot(df['concurrency'], df['ctx_invol'], color='brown', marker='v', label='Involuntary (CPU Saturation)')
    ax3.legend()
    ax3.set_title("3. Context Switches (Lock Contention Evidence)", fontsize=14)
    ax3.set_xlabel('Concurrency', fontsize=12)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(cfg.PLOT_FILE)
    print(f"[Save] Chart -> {cfg.PLOT_FILE}")

if __name__ == "__main__":
    if not os.path.exists(cfg.SERVER_BIN_PATH):
        print(f"Error: Binary not found at {cfg.SERVER_BIN_PATH}")
        exit(1)
    run_benchmark()