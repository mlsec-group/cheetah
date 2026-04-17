import atexit
import signal

import torch
from transformers import TrainerCallback


class TorchProfilerCallback(TrainerCallback):
    def __init__(self, profile_dir="tb_profile", wait=1, warmup=1, active=3, repeat=1):
        self.profile_dir = profile_dir
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.profiler = None

        # Ensure profiler cleanup on exit
        atexit.register(self._safe_exit)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print(f"\n[Profiler] Caught signal {sig}, saving trace...")
        self._safe_exit()
        exit(0)

    def _safe_exit(self):
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            print(f"[Profiler] Trace written to: {self.profile_dir}")
            self.profiler = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
            record_shapes=True,
            with_stack=True,
            with_flops=True,
            profile_memory=True
        )
        self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        self._safe_exit()
