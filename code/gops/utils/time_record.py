import time

class Timer:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = (self.end_time - self.start_time) * 1000
        print(f"Execution time: {elapsed_time:.3f}ms")
