import numpy as np
import torch
import torch.nn as nn

import loaddata
from loaddata import *



def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def metrics_diff_degree(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0

    user_num1 = 0
    all_recall1 = 0
    all_ndcg1 = 0

    user_num2 = 0
    all_recall2 = 0
    all_ndcg2 = 0

    user_num3 = 0
    all_recall3 = 0
    all_ndcg3 = 0

    user_num4 = 0
    all_recall4 = 0
    all_ndcg4 = 0

    train_user_set_p = loaddata.train_user_set_p
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
            if 0<len(train_user_set_p[uid])<=15:
                all_recall1 += hit/len(label)
                all_ndcg1 += dcg/idcg
                user_num1 += 1
            elif 15<len(train_user_set_p[uid])<=30:
                all_recall2 += hit/len(label)
                all_ndcg2 += dcg/idcg
                user_num2 += 1
            elif 30<len(train_user_set_p[uid])<=45:
                all_recall3 += hit/len(label)
                all_ndcg3 += dcg/idcg
                user_num3 += 1
            elif len(train_user_set_p[uid])>45:
                all_recall4 = hit/len(label)
                all_ndcg4 = dcg/idcg
                user_num4 += 1
    # return all_recall / user_num, all_ndcg / user_num
    return all_recall/user_num, all_ndcg/user_num, all_recall1, all_ndcg1,\
           all_recall2, all_ndcg2, all_recall3, all_ndcg3, all_recall4, all_ndcg4,\
           user_num1, user_num2, user_num3, user_num4

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb,device):
    # sp = sp.coalesce().cuda(torch.device(device))
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs = emb[cols] * torch.unsqueeze(sp.values(), dim=1)
    # col_segs = emb[cols] * torch.unsqueeze(sp.values(),dim=1).cuda(torch.device(device))
    result = torch.zeros((sp.shape[0],emb.shape[1]))
    result.index_add_(0, rows, col_segs)
    return result
