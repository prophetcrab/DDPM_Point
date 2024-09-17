from DataUtils.DataSetUtils import *
from DataUtils.DataProcessUtils import *

if __name__ == '__main__':

    file_path = r'D:\PythonProject2\DDPM_Point\dataList.txt'

    batch_size = 32

    dataset = PointDataSet(file_path_list=file_path)

    collate_fn = get_collate_fn()

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=True)

    for batch in dataloader:
        print(batch.shape)




