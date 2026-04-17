from hashlib import sha256

from joblib import Parallel, delayed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn


class ParallelProgress:
    def __init__(self, n_jobs: int, description: str = None, task: str = None):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            refresh_per_second=10
        )
        self.n_jobs = n_jobs
        self.task = task
        task = f" [bold][magenta]{self.task.strip()}[/magenta][/bold] " if self.task else " "
        self.description = description or f"[green]Processing{task}(n_jobs={self.n_jobs}) - "

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception occurred: {exc_type}, {exc_value}")
        return False

    def run(self, func, arg_list, **kwargs):
        iterator = (delayed(func)(*args, **kwargs) for args in arg_list)
        parallel_gen = Parallel(n_jobs=self.n_jobs, return_as="generator")(iterator)
        with self.progress as p:
            task = p.add_task(self.description, total=len(arg_list))
            for item in parallel_gen:
                yield item
                p.advance(task)


def code_hash(code: str):
    return sha256(code.encode()).hexdigest()
