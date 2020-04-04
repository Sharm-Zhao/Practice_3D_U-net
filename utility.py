from time import time



import os
import numpy as np
import SimpleITK as sitk




def sureDir(path):
    # only make new dir when not existing
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

def data_block():
    #原数据120*480*480
    size = np.array([96, 96, 96])  # 取样的slice数量(patch厚度）
    stride =np.array([8,32,32])  # 取样的步长（4*13*13）

    # 用来记录产生的数据的序号
    file_index = 0

    # 用来统计最终剩下的slice数量
    left_slice_list = []

    start_time = time()
    for file in os.listdir("C:/Users/Sharm/Desktop/TF/cyk/dataset/rawdata/imagesTr/"):

        rowdata = sitk.ReadImage(os.path.join("C:/Users/Sharm/Desktop/TF/cyk/dataset/rawdata/imagesTr/", file), sitk.sitkInt16)
        rowdata_array = sitk.GetArrayFromImage(rowdata)

        # seg = sitk.ReadImage(os.path.join("C:/Users/Sharm/Desktop/TF/cyk/dataset/rawdata/labelsTr/", file.replace('data', 'seg')), sitk.sitkInt16)
        seg = sitk.ReadImage(os.path.join("C:/Users/Sharm/Desktop/TF/cyk/dataset/rawdata/labelsTr/", file), sitk.sitkInt16)
        seg_array = sitk.GetArrayFromImage(seg)

        seg_array[seg_array > 0] = 1

        # 在轴向上按照一定的步长进行切块取样，并将结果保存为nii数据
        start_slice = np.array([0, 0, 0])
        end_slice = start_slice + size - 1
        while end_slice[0] <= rowdata_array.shape[0] - 1:
            while end_slice[1] <= rowdata_array.shape[1] - 1:
                while end_slice[2] <= rowdata_array.shape[2] - 1:
                    new_data_array = rowdata_array[start_slice[0]:end_slice[0] + 1, start_slice[1]:end_slice[1] + 1, start_slice[2]:end_slice[2] + 1]
                    new_seg_array = seg_array[start_slice[0]:end_slice[0] + 1, start_slice[1]:end_slice[1] + 1, start_slice[2]:end_slice[2] + 1]

                    new_mra = sitk.GetImageFromArray(new_data_array)
                    new_mra.SetDirection(rowdata.GetDirection())
                    new_mra.SetOrigin(rowdata.GetOrigin())
                    new_mra.SetSpacing(rowdata.GetSpacing())

                    new_seg = sitk.GetImageFromArray(new_seg_array)
                    new_seg.SetDirection(rowdata.GetDirection())
                    new_seg.SetOrigin(rowdata.GetOrigin())
                    new_seg.SetSpacing(rowdata.GetSpacing())


                    new_mra_name = 'data-' + str(file_index) + '.nii'
                    new_seg_name = 'seg-' + str(file_index) + '.nii'

                    sitk.WriteImage(new_mra, os.path.join("C:/Users/Sharm/Desktop/TF/cyk/dataset/after_slice/imagesTr/", new_mra_name))
                    sitk.WriteImage(new_seg, os.path.join("C:/Users/Sharm/Desktop/TF/cyk/dataset/after_slice/labelsTr/", new_seg_name))

                    file_index += 1

                    start_slice[2] += stride[2]
                    end_slice[2] = start_slice[2] + size[2] - 1

                start_slice[2] = 0
                end_slice[2] = start_slice[2] + size[2] - 1
                start_slice[1] += stride[1]
                end_slice[1] = start_slice[1] + size[1] - 1

            start_slice[2] = 0
            end_slice[2] = start_slice[2] + size[2] - 1
            start_slice[1] = 0
            end_slice[1] = start_slice[1] + size[1] - 1
            start_slice[0] += stride[0]
            end_slice[0] = start_slice[0] + size[0] - 1
        print(file_index)
        print('already use {:.3f} min'.format((time() - start_time) / 60))
        print('-----------')