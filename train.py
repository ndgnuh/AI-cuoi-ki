from ptrseg.utils import SegmentDataset, DataLoader

ds = SegmentDataset(root="data/PPM-100/", mask_dir="matte")

for epoch in range(2):
    print("---")
    for (image, mask) in DataLoader(ds, batch_size=4, num_workers=4):
        print(image.shape, mask.shape)
