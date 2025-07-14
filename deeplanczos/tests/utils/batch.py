from itertools import islice

class batch_iterator:
    def __init__(self, sample_input, sample_labels, batch_size=2, imgs=True):
        self.sample_input = sample_input
        self.sample_labels = sample_labels
        self.batch_size = batch_size
        self.imgs = imgs

    def __iter__(self):
        for i in range(0, len(self.sample_input), self.batch_size):
            if self.imgs:
                batch = {
                    'imgs': self.sample_input[i:i + self.batch_size],
                    'labels': self.sample_labels[i:i + self.batch_size]
                }
            else:
                batch = self.sample_input[i:i + self.batch_size], self.sample_labels[i:i + self.batch_size]
            yield batch