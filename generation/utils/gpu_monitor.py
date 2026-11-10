import gc,time
from contextlib import contextmanager
import torch

class GPUMonitor:
    def __init__(self): self.records=[]; self.enabled=torch.cuda.is_available(); self.peak_bytes=0
    @contextmanager
    def measure(self,label):
        if self.enabled: torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter(); record={'label':label}
        try: yield record
        finally:
            if self.enabled:
                torch.cuda.synchronize(); peak=torch.cuda.max_memory_allocated(); self.peak_bytes=max(self.peak_bytes,peak)
                record.update(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),peak_allocated_bytes=peak)
            record['seconds']=time.perf_counter()-started; self.records.append(record)
    def clear(self):
        gc.collect()
        if self.enabled: torch.cuda.empty_cache()
