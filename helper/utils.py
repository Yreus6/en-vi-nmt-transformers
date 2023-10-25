import os
from tqdm import tqdm
import multiprocessing as mp


class SaveHandle(object):
    """handle the number of """

    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


class MultiProcessorWriter:

    def __init__(self, output_files, process_fn, listener_fn, data_len, batch_size):
        manager = mp.Manager()
        self.output_files = output_files
        self.process_fn = process_fn
        self.listener_fn = listener_fn
        self.data_len = data_len
        self.batch_size = batch_size
        self.q = manager.Queue()

    def worker(self, idx):
        batch_res = self.process_fn(idx, self.batch_size, self.data_len)
        self.q.put(batch_res)

    def listener(self):
        """listens for messages on the q"""
        while 1:
            m = self.q.get()
            if m == 'kill':
                break

            self.listener_fn(m, self.output_files)

    def run(self):
        pool = mp.Pool(mp.cpu_count() + 2)
        # put listener to work first
        watcher = pool.apply_async(self.listener)

        # fire off workers
        jobs = []
        for i in range(0, self.data_len, self.batch_size):
            job = pool.apply_async(self.worker, (i,))
            jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in tqdm(jobs):
            job.get()

        # now we are done, kill the listener
        self.q.put('kill')
        watcher.get()
        pool.close()
        pool.join()
