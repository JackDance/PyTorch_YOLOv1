from torch.utils.data import Dataset,DataLoader
import numpy as np


# 构造数据集
class TxtDataset(Dataset):
    def __init__(self):
        self.data = np.asarray([[1,2], [3,4], [5,6], [7,8], [9,10], [11, 12]])
        self.label = np.asarray([1,2,0,1, 1, 2])

    def __getitem__(self, item): # item is index
        data = self.data[item]
        label = self.label[item]
        return data, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    # 实例化TxtDataset
    Txt = TxtDataset()
    # 获得一个数据
    print(Txt[1]) # (array([3, 4]), 2)
    # 获得数据集的长度
    print(Txt.__len__()) # 4

    # 实例化Dataloader
    test_loader = DataLoader(Txt, batch_size=2, shuffle=True, num_workers=2)

    for i , train_data in enumerate(test_loader):
        print('i: ', i)
        data, label = train_data
        print('data: ', data)
        print('label: ', label)

        # i:  0
        # data:  tensor([[ 9, 10],
        #         [ 5,  6]])
        # label:  tensor([1, 0])
        # i:  1
        # data:  tensor([[7, 8],
        #         [3, 4]])
        # label:  tensor([1, 2])
        # i:  2
        # data:  tensor([[ 1,  2],
        #         [11, 12]])
        # label:  tensor([1, 2])