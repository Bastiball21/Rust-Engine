import subprocess
import time
import sys
import threading
import queue

ENGINE_PATH = "./target/release/aether"

def read_output(process, q):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        q.put(line.strip())

def run_bench():
    print("Running baseline benchmark...")
    # Keep input open to prevent early exit
    # We send command then sleep then close
    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q = queue.Queue()
    t = threading.Thread(target=read_output, args=(proc, q))
    t.daemon = True
    t.start()

    try:
        proc.stdin.write("uci\n")
        proc.stdin.write("isready\n")
        proc.stdin.write("position startpos\n")
        proc.stdin.write("go depth 14\n")
        proc.stdin.flush()

        start_time = time.time()
        nodes = 0
        bestmove_found = False

        while True:
            try:
                line = q.get(timeout=30) # 30s timeout
                # print(f"Engine: {line}")
                if "nodes" in line and "nps" in line:
                    parts = line.split()
                    try:
                        n_idx = parts.index("nodes")
                        nodes = int(parts[n_idx+1])
                    except:
                        pass
                if line.startswith("bestmove"):
                    bestmove_found = True
                    break
            except queue.Empty:
                print("Timeout waiting for engine output")
                break

        end_time = time.time()
        duration = end_time - start_time

        if bestmove_found:
            nps = nodes / duration
            print(f"Benchmark: Depth 14 found in {duration:.3f}s, Nodes: {nodes}, NPS: {nps:.0f}")
        else:
            print("Benchmark Failed: No bestmove found")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        proc.terminate()

if __name__ == "__main__":
    run_bench()
