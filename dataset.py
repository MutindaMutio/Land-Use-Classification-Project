class EuroSATDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
if __name__ == "__main__":
    ds = EuroSATDataset([10, 20, 30], [0, 1, 2])
    print(ds[1])