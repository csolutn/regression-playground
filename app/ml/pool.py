"""Runs trainings in worker processes, so a class of students training at once do not fight over one GIL.

Measured on a 12-core Mac: 8 small runs at once took 15 s each in threads, 1 s each in processes.
The web server thread only relays events: ``TrainingPool.stream(cfg)`` yields the trainer's events.
"""
import multiprocessing as mp
import queue
import threading

from concurrent.futures import ProcessPoolExecutor

from . import trainer


def _run(cfg, out, stop, time_limit):
    """In a worker process: send each event to ``out``, then None. Ends early when ``stop`` is set."""
    try:
        for event in trainer.train(cfg, time_limit=time_limit):
            if stop.is_set():
                return
            out.put(event)
    except Exception as exc:          # sent to the web server, which reports it
        out.put({'type': 'crash', 'error': repr(exc)})
    finally:
        out.put(None)


class TrainingPool:
    def __init__(self, size):
        ctx = mp.get_context('spawn')            # fork is unsafe with torch threads
        self.slots = threading.BoundedSemaphore(size)
        self.executor = ProcessPoolExecutor(size, mp_context=ctx)
        self.manager = ctx.Manager()

    def acquire(self, timeout):
        return self.slots.acquire(timeout=timeout)

    def stream(self, cfg, time_limit, idle_timeout):
        """Yield the run's events. Call after acquire(); the slot is freed when the worker finishes."""
        out, stop = self.manager.Queue(), self.manager.Event()
        try:
            future = self.executor.submit(_run, cfg, out, stop, time_limit)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        try:
            while (event := out.get(timeout=idle_timeout)) is not None:
                if event['type'] == 'crash':
                    raise RuntimeError(event['error'])
                yield event
        except queue.Empty:
            raise TimeoutError('the training worker stopped answering') from None
        finally:
            stop.set()                  # the browser left or something failed: stop the worker too
