"""
run_pipeline.py

Runs three pipeline scripts in sequence:
  1. Normal Seg Methods/Normalizedmrf_cl.py
  2. Normal Seg Methods/Normalizededge_cl.py
  3. DL Methods/DeepLabV3+/train_lung_seg.py

Stops immediately if a script exits with an error (non-zero return code).
Set STOP_ON_FAILURE = False below if you'd rather continue to the next
script even after a failure.
"""

import subprocess
import sys
import time

STOP_ON_FAILURE = True

SCRIPTS = [
    r"Normal Seg Methods\Normalizedmrf_cl.py",
    r"Normal Seg Methods\Normalizededge_cl.py",
    r"DL Methods\DeepLabV3+\train_lung_seg.py",
]


def run_script(path):
    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {path}")
    print(f"{'=' * 60}\n")

    start = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n[OK] {path} finished in {elapsed:.1f}s")
        return True
    else:
        print(f"\n[FAILED] {path} exited with code {result.returncode} "
              f"after {elapsed:.1f}s")
        return False


def main():
    overall_start = time.time()
    results = {}

    for script in SCRIPTS:
        success = run_script(script)
        results[script] = success

        if not success and STOP_ON_FAILURE:
            print("\nStopping pipeline due to failure (STOP_ON_FAILURE=True).")
            break

    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {script}")
    print(f"\nTotal time: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()