import pandas as pd
import numpy as np

# 向文件追加写入数据
def write_file(filename, lines):
    with open(filename, 'at') as f:
        f.writelines(lines)

# 返回向量角度结果组
def get_angle(vector_mat, vector_arr):
    vector_mat_norm = np.linalg.norm(vector_mat,axis=1)
    vector_arr_norm = np.linalg.norm(vector_arr)

    mat_dot_arr = vector_mat.dot(vector_arr)
    cos_value = mat_dot_arr / (vector_mat_norm * vector_arr_norm)
    # 保留6位小数点的精度，以防出现cos值超过[-1,1]的范围
    cos_value=cos_value.map(lambda x:float("%.6f" % x))
    # 返回经过带有索引的排序的角度结果组（索引为MD5值）
    angle_ser=cos_value.map(lambda x: np.rad2deg(np.arccos(x))).sort_values()
    return angle_ser

# xvector向量格式转化保存为容易矩阵读取
def vector_tran(init_vector_file,save_vector_file):
    with open(init_vector_file, 'r') as file:
        data = file.readlines()
        vector_list = []
        for row in data:
            rowdata = row.strip().replace('[', '').replace(']', '').split('  ')
            label = rowdata[0].split('-')[-1]
            # vector=np.array(rowdata[1].strip().split(), dtype=float)
            vector = rowdata[1].strip()
            vector_list.append(label + ' ')
            vector_list.append(vector + '\n')
    write_file(save_vector_file, vector_list)

# 把向量格式转换并保存
# vector_tran(r'C:\Users\johny\Desktop\50_train_xvec.txt',r'C:\Users\johny\Desktop\50_train_xvec_db.txt')
# vector_tran(r'C:\Users\johny\Desktop\50_test_xvec.txt',r'C:\Users\johny\Desktop\50_test_xvec_db.txt')

# 读取全体注册向量表
xvector_db = pd.read_table(r'C:\Users\johny\Desktop\50_train_xvec_db.txt', header=None, sep=' ',index_col=0)
# print(xvector_db)

# 读取待识别向量表
xvector_rec = pd.read_table(r'C:\Users\johny\Desktop\50_test_xvec_db.txt', header=None, sep=' ',index_col=0)
keys=[]
values=[]
for file_name, row in xvector_rec.iterrows():
    #print(row)

    # 返回待识别音频文件向量与全体注册向量的匹配结果（角度矩阵）
    angle_mat=pd.DataFrame(get_angle(xvector_db,row))
    # print(angle_mat)
    angle_mat.rename(columns={0:'angle'},inplace=True)
    angle_mat.reset_index(inplace=True)
    if len(angle_mat[angle_mat.angle < 40])>0:
        result=angle_mat[angle_mat.angle < 40][0].value_counts()
    elif len(angle_mat[angle_mat.angle < 45])>0:
        result = angle_mat[angle_mat.angle < 45][0].value_counts()
    elif len(angle_mat[angle_mat.angle < 50])>0:
        result = angle_mat[angle_mat.angle < 50][0].value_counts()
    else:
        result = angle_mat[angle_mat.angle < 70][0].value_counts()
    #print(file_name,result.index[0])
    #print(file_name,result)

    keys.append(file_name)
    values.append(result.index[0])
    #break
result_df=pd.DataFrame()
result_df['keys']=keys
result_df['values']=values
print(result_df)
result_df.to_csv(r'C:\Users\johny\Desktop\50_result.csv',index=False,sep=',')
