"""
创建人：MECH
创建时间：2022/02/01
功能描述：faiss索引构建检索系统的全流程
"""
import time

from typing import List, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
import faiss

from pandas import DataFrame
from numpy import array, ndarray
import pickle


class FaissSearcher:
    """
    faiss索引构建检索系统的全流程
    """
    def __init__(
            self,
            vecs,
            measurement: str = None,
            norm_vec: bool = False,
            index_param: str = None,
            **kwargs
    ):

        self.index_param = index_param
        self.items = vecs
        self.norm_vec = True if measurement == 'cos' else norm_vec
        self.metric = self.set_measure_metric(measurement)
        self.measurement = measurement
        self.vec_dim = vecs.shape[1]
        self.vecs = vecs
        self.index = None
        self.kwargs = kwargs

    @staticmethod
    def set_measure_metric(measurement):
        metric_dict = {
            'cos': faiss.METRIC_INNER_PRODUCT,
            'l1': faiss.METRIC_L1,
            'l2': faiss.METRIC_L2,
            'l_inf': faiss.METRIC_Linf,
            'l_p': faiss.METRIC_Lp,
            'brayCurtis': faiss.METRIC_BrayCurtis,
            'canberra': faiss.METRIC_Canberra,
            'jensen_shannon': faiss.METRIC_JensenShannon
        }
        if measurement in metric_dict:
            return metric_dict[measurement]
        else:
            raise Exception(f"Do not support measurement: '{measurement}', support measurement is [{', '.join(list(metric_dict.keys()))}]")

    @staticmethod
    def __tofloat32__(vecs):
        return vecs.astype(np.float32)

    @staticmethod
    def __normvec__(vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def __build_faiss_index(self):
        if 'hnsw' in self.index_param.lower() and ',' not in self.index_param:
            self.index = faiss.IndexHNSWFlat(self.vec_dim, int(self.index_param.split('HNSW')[-1]), self.metric)
        else:
            self.index = faiss.index_factory(self.vec_dim, self.index_param, self.metric)
        self.index.verbose = True
        self.index.do_polysemous_training = False
        return self

    def load_index(self, index_path):
        print(f"Load index...")
        self.index = faiss.read_index(index_path)
        assert self.index.ntotal == len(self.vecs), f"Index sample nums {self.index.ntotal} != Items length {len(self.vecs)}"
        assert self.index.d == self.vec_dim, f"Index dim {self.index.d} != Vecs dim {self.vec_dim}"
        assert self.index.is_trained, "Index dose not trained"

    def train(self):
        print(f"Encode items start...")
        start_time = time.time()
        # vecs = self.__tofloat32__(self.vecs)
        print(f"Train index start...")
        self.__build_faiss_index()
        self.index.train(self.vecs)
        self.index.add(self.vecs)
        print(f"Train index cost time: {time.time() - start_time}")

    def search_items(self, target: List[str], indexes: ndarray, directories: ndarray, keep_rank_no: bool = False) -> \
            Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray], DataFrame]:
        """
        返回df列名：["source_item", "sim_item", "sim_val" + 原items除第一列外的剩下的列]
        """
        start_time = time.time()
        if not self.encoder and keep_rank_no:
            res = (self.item_list[indexes], directories[indexes], indexes)
        elif not self.encoder and not keep_rank_no:
            res = (self.item_list[indexes], directories)
        else:
            target = pd.DataFrame(target, columns=['source_item'])
            target['sim_ind'] = list(indexes)
            target['sim_val'] = list(directories)
            target['pair'] = target.apply(lambda x: [[c[0], c[1], i] for i, c in enumerate(zip(x['sim_ind'], x['sim_val']))], axis=1)
            target = target.drop(columns=['sim_ind', 'sim_val'])
            target = target.explode('pair').reset_index(drop=True)
            target[['sim_ind', 'sim_val', 'rank_no']] = pd.DataFrame(target['pair'].to_list(), columns=['sim_ind', 'sim_val', 'rank_no'])
            target['sim_val'] = target['sim_val'].values.astype(np.float32)
            sim_item = self.items.iloc[target['sim_ind']].reset_index(drop=True)
            sim_item.columns = ['sim_item'] + list(sim_item.columns[1:])
            target = target.drop(columns=['pair', 'sim_ind']) if keep_rank_no else target.drop(columns=['pair', 'sim_ind', 'rank_no'])

            if self.query_feature_sep:
                print(f"[Warning] Out Put Query Will Be Split By '{self.query_feature_sep}'")
                target["source_item"] = target["source_item"].apply(lambda x: str(x).split(self.query_feature_sep)[0])

            if self.doc_feature_sep:
                print(f"[Warning] Out Put Document Will Be Split By '{self.doc_feature_sep}'")
                sim_item["sim_item"] = sim_item["sim_item"].apply(lambda x: str(x).split(self.doc_feature_sep)[0])
            res = pd.concat([target, sim_item], axis=1)
        print(f"Find items cost time: {time.time() - start_time}")
        return res

    def search(self, target, topK: Union[int, List[int]], keep_rank_no=False):
        if self.index:
            if isinstance(topK, int):
                start_time = time.time()
                directories, indexes = self.index.search(target, topK)
                # print(f"Search index cost time: {(time.time() - start_time) * 1000}")
                return directories, indexes
                # return self.search_items(target, indexes, directories, keep_rank_no=keep_rank_no)

    def save_index(self, index_save_path):
        faiss.write_index(self.index, index_save_path)

    def cal_sim(self, item1: str, items2: List[str]):
        vec1 = self.encoder.encode([item1], verbose=0)
        vecs2 = self.encoder.encode(items2, verbose=0)
        sim_score_list = list(vec1.dot(vecs2.T))
        sim_df = pd.DataFrame([items2], columns=['item'])
        sim_df['score'] = sim_score_list
        return sim_df.sort_values(by="score", ascending=False)

    def save_searcher(self, path):
        file = open(path, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_searcher(path):
        file = open(path, "rb")
        return pickle.load(file)
