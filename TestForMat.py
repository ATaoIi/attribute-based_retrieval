import pickle

# 指定pkl文件的路径
# pkl_file_path = 'data/PETA/dataset_all.pkl'
#
# dataset_info = pickle.load(open(pkl_file_path, 'rb+'))
#
# # 输出内容到控制台
# print(dataset_info.root)
# # root': '/mnt/data1/jiajian/datasets/attribute/PETA/images',
#
#
# # 以读取二进制模式打开文件
# with open(pkl_file_path, 'rb') as f:
#     # 使用pickle库加载文件内容
#     data = pickle.load(f)
#
# # 修改字典中'root'键的值
# data['root'] = r'C:\Users\licha\Desktop\行人分析数据集\upar_challenge-main\data\PETA\images'
#
# # 以写入二进制模式打开文件
# with open(pkl_file_path, 'wb') as f:
#     # 使用pickle库将修改后的数据写回文件
#     pickle.dump(data, f)
#
# dataset_info = pickle.load(open(pkl_file_path, 'rb+'))
#
# # 输出内容到控制台
# print(dataset_info.root)