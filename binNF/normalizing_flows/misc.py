from tqdm.autonotebook import tqdm

class tqdm_recycled(tqdm):

    def close(self):
        self.reset()

    def really_close(self):
        try:
            self.sp(close=True)
        except (AttributeError, TypeError) as e:
            pass