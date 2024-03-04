# from mpi4py import MPI
# import sys
# import pandas as pd
# import os
# import requests
#
#
# url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"
# response = requests.get(url)
#
# with open("a1a_dataset", "wb") as file:
#     file.write(response.content)
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
#
# input_file = "a1a_dataset"
# comm.Barrier()
# # 读取数据集文件
# df = pd.read_csv(input_file, header=None, delimiter=" ")
#
# # 计算每个进程要处理的行数
# total_rows = len(df)
# rows_per_process = total_rows // size
# start_row = rank * rows_per_process
# end_row = start_row + rows_per_process
#
# # 如果是最后一个进程，处理剩余的行数
# if rank == size - 1:
#     end_row = total_rows
#
# # 切分数据集文件
# chunk_df = df.iloc[start_row:end_row]
#
# # 创建文件夹
# folder_name = str(size)
# if rank == 0:
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#
# # 将切分的文件放入文件夹
# output_file = os.path.join(folder_name, f"{os.path.splitext(os.path.basename(input_file))[0]}_part_{rank+1}")
# chunk_df.to_csv(output_file, index=False, header=False, sep=" ")
# comm.Barrier()
# # print(f"Process {rank} completed.")
# # comm = MPI.COMM_WORLD
# # rank = comm.Get_rank()
# # size = comm.Get_size()
# # print('My rank is ',rank)
# # x = ho.sayhello(comm)
# import train_framework as ho
# ho.test_main(comm)
# # ho.test_main2(comm)
# # try:
# #     ho.test_main(list())
# # except:
# #     pass
# # else:
# #     assert 0, "exception not raised"
# # MPI.Finalize()
# # print("rank is %d, x= %d"%(rank,x))
# #
# ## ----- Object creation -----
# #
#
import train_framework as ho
from mpi4py import MPI
import pandas as pd
import os
import requests

def process_data():
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"
    response = requests.get(url)

    with open("a1a_dataset", "wb") as file:
        file.write(response.content)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_file = "a1a_dataset"
    comm.Barrier()
    # 读取数据集文件
    df = pd.read_csv(input_file, header=None, delimiter=" ")

    # 计算每个进程要处理的行数
    total_rows = len(df)
    rows_per_process = total_rows // (size-1)
    start_row = rank * rows_per_process
    end_row = start_row + rows_per_process

    # 如果是最后一个进程，处理剩余的行数
    if rank == size - 1:
        end_row = total_rows

    # 切分数据集文件
    chunk_df = df.iloc[start_row:end_row]

    # 创建文件夹
    folder_name = str(size)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 将切分的文件放入文件夹
    output_file = os.path.join(folder_name, f"{os.path.splitext(os.path.basename(input_file))[0]}_part_{rank+1}")
    chunk_df.to_csv(output_file, index=False, header=False, sep=" ")
    comm.Barrier()

# 先运行process_data()函数
process_data()

# 然后运行ho.test_main(comm)

comm = MPI.COMM_WORLD
ho.test_main(comm)

