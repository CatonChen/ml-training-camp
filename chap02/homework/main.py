# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm


def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result


def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


def call_target_mean_v1(data, y_name, x_name):
    start_time = time.time()
    result = target_mean_v1(data, y_name, x_name)
    end_time = time.time()
    print('原始实现target_mean_v1执行时间： ', end_time - start_time)


def call_target_mean_v2(data, y_name, x_name):
    start_time = time.time()
    result = target_mean_v2(data, y_name, x_name)
    end_time = time.time()
    print('原始实现target_mean_v2执行时间： ', end_time - start_time)


def call_tm_target_mean_v3(data, y_name, x_name):
    start_time = time.time()
    result = tm.target_mean_v3(data, y_name, x_name)
    end_time = time.time()
    print('cython实现tm.target_mean_v3执行时间： ', end_time - start_time)


def call_tm_target_mean_v4(data, y_name, x_name):
    start_time = time.time()
    result = tm.target_mean_v4(data, y_name, x_name)
    end_time = time.time()
    print('cython实现tm.target_mean_v4执行时间： ', end_time - start_time)


def main():
    # 5000样本
    print('5000观测样本：')
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    call_target_mean_v1(data, 'y', 'x')
    call_target_mean_v2(data, 'y', 'x')
    call_tm_target_mean_v3(data, 'y', 'x')
    call_tm_target_mean_v4(data, 'y', 'x')
    # 100000样本
    print('100000观测样本：')
    y2 = np.random.randint(2, size=(100000, 1))
    x2 = np.random.randint(10, size=(100000, 1))
    data2 = pd.DataFrame(np.concatenate([y2, x2], axis=1), columns=['y', 'x'])
    # call_target_mean_v1(data2, 'y', 'x')
    call_target_mean_v2(data2, 'y', 'x')
    call_tm_target_mean_v3(data2, 'y', 'x')
    call_tm_target_mean_v4(data2, 'y', 'x')

    # start_time = time.time()
    # result_1 = target_mean_v1(data, 'y', 'x')
    # result_2 = target_mean_v2(data, 'y', 'x')
    # result_3 = tm.target_mean_v3(data, 'y', 'x')
    # diff = np.linalg.norm(result_1 - result_2)
    # print(diff)
    # end_time = time.time()
    # print(end_time - start_time)


if __name__ == '__main__':
    main()
