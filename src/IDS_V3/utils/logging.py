import time

class Timer:
    def __init__(self, name=""):
        self.name = name
    def __enter__(self):
        self.t = time.time()
        if self.name:
            print(f"\n--- {self.name} ---")
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.t
        if self.name:
            print(f"  ({elapsed:.1f}s)")
