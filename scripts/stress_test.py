import subprocess
import time
import sys
import threading
import queue
import random

ENGINE_PATH = "./target/release/aether"

def stress_scenario_a():
    print("\n--- Running Stress Scenario A: Concurrent GO commands (Race Check) ---")
    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q = queue.Queue()
    def reader():
        while True:
            line = proc.stdout.readline()
            if not line: break
            q.put(line.strip())

    t_reader = threading.Thread(target=reader)
    t_reader.daemon = True
    t_reader.start()

    # Initialize
    proc.stdin.write("uci\n")
    proc.stdin.write("isready\n")
    proc.stdin.write("position startpos\n")
    proc.stdin.flush()

    time.sleep(1)

    errors = []

    def spammer(id):
        for _ in range(10):
            try:
                # Randomly choose between infinite, depth, nodes
                cmd_type = random.choice(["depth 5", "nodes 1000", "movetime 100"])
                cmd = f"go {cmd_type}\n"
                proc.stdin.write(cmd)
                proc.stdin.flush()
                time.sleep(random.uniform(0.01, 0.1)) # Fast overlapping commands

                # Stop sometimes
                if random.random() < 0.3:
                    proc.stdin.write("stop\n")
                    proc.stdin.flush()
            except Exception as e:
                errors.append(e)

    threads = []
    for i in range(8):
        t = threading.Thread(target=spammer, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check if engine is still alive
    if proc.poll() is not None:
        print("FAIL: Engine crashed during stress test A")
        return False

    proc.terminate()
    if errors:
        print(f"FAIL: Exceptions in spammer threads: {errors}")
        return False

    print("PASS: Scenario A completed without crash")
    return True

def stress_scenario_c():
    print("\n--- Running Stress Scenario C: Threads=8 Stability ---")
    proc = subprocess.Popen(
        [ENGINE_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    q = queue.Queue()
    def reader():
        while True:
            line = proc.stdout.readline()
            if not line: break
            q.put(line.strip())

    t_reader = threading.Thread(target=reader)
    t_reader.daemon = True
    t_reader.start()

    try:
        proc.stdin.write("uci\n")
        proc.stdin.write("setoption name Threads value 8\n")
        proc.stdin.write("isready\n")
        proc.stdin.write("position startpos\n")
        proc.stdin.write("go movetime 3000\n") # Run for 3 seconds
        proc.stdin.flush()

        start = time.time()
        active = True
        while time.time() - start < 5:
            if proc.poll() is not None:
                print("FAIL: Engine crashed")
                return False
            time.sleep(0.1)

        # Stop explicitly if not finished
        proc.stdin.write("stop\n")
        proc.stdin.flush()
        time.sleep(1)

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        proc.terminate()

    print("PASS: Scenario C completed")
    return True

if __name__ == "__main__":
    res_a = stress_scenario_a()
    res_c = stress_scenario_c()

    if res_a and res_c:
        print("\nALL STRESS TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSTRESS TESTS FAILED")
        sys.exit(1)
