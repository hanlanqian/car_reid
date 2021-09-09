import numpy as np
import torch
import pickle
import argparse
import matplotlib.pyplot as plt


def img_show(img_query_list: list, img_result_list: list, save: bool = False):
    query_len = len(img_query_list)
    result_len = len(img_result_list)
    ax = plt.figure(dpi=500)
    plt.axis('off')
    plt.tight_layout(2.4)
    count = 1
    for i in range(query_len):
        for j in range(result_len + 1):
            plt.subplot(query_len, result_len + 1, count)
            count += 1
            if not (i or j):
                plt.title('Query')
            if j == 0:
                img = plt.imread(img_query_list[i])
                plt.axis('off')
                plt.imshow(img)
            else:
                plt.axis('off')
                plt.imshow(plt.imread(img_result_list[j - 1][i]))
    plt.show()
    if save:
        plt.imsave()


def N_rank(rank_num, pkl_path):
    image_path = []
    image_query_path = []
    with open(pkl_path, 'rb') as f:
        file = torch.load(f)

    # _index = np.sort(file['distmat'], axis=0)[:rank_num]
    _index = file['distmat'].argpartition(rank_num, axis=1)

    for j in range(rank_num):
        image_path.append([file['paths'][i + len(_index)] for i in _index[:, j]])
        image_query_path = file['paths'][:len(_index)]
    return image_path, image_query_path


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--pkl-path', type=str, default='./outputs/my_outputs/test_output.pkl')
    parse.add_argument('--rank-num', type=int, default=5)
    parse.add_argument('--plot', type=bool, default=True)
    args = parse.parse_args()

    image_path, image_query_path = N_rank(args.rank_num, args.pkl_path)
    if args.plot:
        img_show(image_query_path, image_path)
    else:
        print(image_query_path)
        print(image_path)
