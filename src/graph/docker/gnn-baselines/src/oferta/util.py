from joblib import Parallel, delayed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, track, MofNCompleteColumn


class ParallelProgress:
    def __init__(self, n_jobs: int, description: str = None, task: str = None):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            refresh_per_second=0.5
        )
        self.n_jobs = n_jobs
        task = f" [bold][magenta]{task.strip()}[/magenta][/bold] " if task else " "
        self.description = description or f"[green]Processing{task}(n_jobs={n_jobs})..."

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):

        if exc_type is not None:
            print(f"An exception occurred: {exc_type}, {exc_value}")

        return False

    def run(self, func, arg_list, **kwargs):
        iterator = (delayed(func)(*args, **kwargs) for args in arg_list)
        parallel_gen = Parallel(n_jobs=self.n_jobs, return_as="generator")(iterator)
        result = list(track(parallel_gen, description=self.description, total=len(arg_list)))
        return result
