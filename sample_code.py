# data loader
from torch.utils.data import Dataset

NUM_ATTR = 6


class FashionNet_Dataset(Dataset):

    def __init__(self, root, txt, dataset):
        self.img_path = []
        self.labels = [[] for _ in range(NUM_ATTR)]

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                # make dummy label for test set
                if 'test' in txt:
                    for i in range(NUM_ATTR):
                        self.labels[i].append(0)
        if 'test' not in txt:
            with open(txt.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(NUM_ATTR):
                        self.labels[i].append(int(attrs[i]))

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):

        path = self.img_path[index]
        label = np.array([self.labels[i][index] for i in range(NUM_ATTR)])

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        ...

        return sample, label, index


# evaluation
def compute_avg_class_acc(gt_labels, pred_labels):
    num_attr = 6
    num_classes = [7, 3, 3, 4, 6, 3]  # number of classes in each attribute

    per_class_acc = []
    for attr_idx in range(num_attr):
        for idx in range(num_classes[attr_idx]):
            target = gt_labels[:, attr_idx]
            pred = pred_labels[:, attr_idx]
            correct = np.sum((target == pred) * (target == idx))
            total = np.sum(target == idx)
            per_class_acc.append(float(correct) / float(total))

    return sum(per_class_acc) / len(per_class_acc)
