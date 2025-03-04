from .._model import DMmodel
from ...util.basicutil import param_default
from ...tensor import Graph, TensorData

import datetime
import numpy as np
from typing import List, Any, Union, Iterable

from scipy import stats

class DenseFlow( DMmodel ):
    def __init__(self, tensor: TensorData, *args, **kwargs):
        """Detect the most density subgraph on Ethereum Transactions for Anti-MonenLundry.

        ## Required TensorData Format:
        | From |  To  | *Datas |
        | :--: | :--: | :----: |
        | int  | int  |   Any  |

        Args:
            tensor (spartan.Graph): The tensor of graph data.
            timeslot (int): The size of the timeslot in day(s), default 7.
            b (int): The hyperparameter for suspiciousness fusion, default 32.
            timeformat (str): The format for time string parser, formating as datetime.datetime.strptime, default '%Y-%m-%d'.
            dt_idx (int): The index of the timestamp col in raw data, default 2.
            value_idx (int): The index of the value col in raw data, default 3.
            value_scaler (float): The scaler applied on value, default 1. 1e-18 for Eth and 1e-6 for Tron if required.
        """
        super().__init__(tensor, *args, **kwargs)
        self.tensor = tensor
        self.timeslot = param_default(kwargs, 'timeslot', 7)
        self.b = param_default(kwargs, 'b', 32)
        self.timeformat = param_default(kwargs, 'timeformat', '%Y-%m-%d')
        self.datetime_index = param_default(kwargs, 'dt_idx', 2)
        self.value_index = param_default(kwargs, 'value_idx', 3)
        self.value_scaler = param_default(kwargs, 'value_scaler', 1)

        assert self.datetime_index >= 2 and self.value_index >= 2 and self.datetime_index != self.value_index, \
            "dt_idx or value_idx cannot less than 2 or equal! The first and second col are requied as From and To id."

        self.graph = TensorData(tensor.data.iloc[:, :2]).toSTensor(hasvalue=False)
        
        adj = self.graph.todense()
        self.nodes_cnt = int(max(*adj.shape))
        self.adj = np.zeros((self.nodes_cnt, self.nodes_cnt), dtype=adj.dtype)
        self.adj[np.where(adj != 0)] = 1
        self.subset_indicator = np.ones(self.nodes_cnt)

        # neighbor index map
        self.search_mat = np.ones((self.nodes_cnt, self.nodes_cnt), dtype=np.int32) * -1
        self.search_list: List[List[Any]] = list()
        self.burst_point_index = None
        self.awakening_point_1st_index = None
        self.delta_c_am_milti_s_am = None
        self.ta_tm_indicator = None
        self.__build_search_index()

        # time related cache
        self.timeslot_base = None
        self.timeslots: List[List[Any]] = list()
        self.timeslot_index = list()
        self.timeorder_datas = list()
        self.timeorder_datas_index_map = None  # index in timeorder list: index in og tensor
        self.timeorder_datas_reverse_index_map = None  # index in og tensor: index in timeorder list
        self.__build_timeslot_sequence()

        # data array
        self.value_array = None
        self.__build_data_array()

        # time-based R list
        self.R_list = None
        self.__build_R_list()

        # update all-nodes susipiciousness
        self.suspiciousness = np.zeros(self.nodes_cnt)
        self.update_suspiciousness()

    def __build_R_list(self):
        result = []
        cur_data_index = 0
        cur_slot_index = 0
        accumulated_data_count = 0  # data count in historical slots
        left_2d_index, right_2d_index = 0, 0
        range_value = []
        slot_value = [i[self.value_index - 2] * self.value_scaler for i in self.timeslots[cur_slot_index]]
        while cur_data_index < len(self.timeorder_datas):
            if cur_data_index - accumulated_data_count >= len(self.timeslots[cur_slot_index]):
                accumulated_data_count += len(self.timeslots[cur_slot_index])
                cur_slot_index += 1
                slot_value = [i[self.value_index - 2] * self.value_scaler for i in self.timeslots[cur_slot_index]]
            datas, cur_timestamp = self.timeorder_datas[cur_data_index]

            # adjust left 2d index scaler
            while True:
                _, target_timestamp = self.timeorder_datas[left_2d_index]
                if cur_timestamp - target_timestamp > 2 * 24 * 60 * 60:
                    left_2d_index += 1
                    range_value.pop(0)
                    continue
                break
            while True:
                if right_2d_index >= len(self.timeorder_datas) - 1:
                    break
                datas, target_timestamp = self.timeorder_datas[right_2d_index]
                if target_timestamp - cur_timestamp < 2 * 24 * 60 * 60:
                    right_2d_index += 1
                    range_value.append(datas[self.value_index - 2] * self.value_scaler)
                    continue
                break
            
            if len(slot_value) == 0 or sum(slot_value) == 0:
                result.append(0)
            else:
                result.append(sum(range_value) / sum(slot_value))
            cur_data_index += 1
        self.R_list = np.array(result, dtype=np.float32)

    def __build_data_array(self):
        value_list = []
        for _, row in self.tensor.data.iterrows():
            _, _, *datas = row
            value = datas[self.value_index - 2] * self.value_scaler
            value_list.append(value)
        self.value_array = np.array([value_list], dtype=np.float32)  # #feature * #edge
    
    def __build_search_index(self):
        current_search_index = len(self.search_list)
        for idx, row in self.tensor.data.iterrows():
            f, t, *datas = row
            if self.search_mat[f, t] == -1:
                self.search_list.append([(datas, idx)])
                self.search_mat[f, t] = current_search_index
                current_search_index += 1
                continue
            # not empty
            last_index = int(self.search_mat[f, t])
            self.search_list[last_index].append((datas, idx))
        assert not ((self.adj == 0) ^ (self.search_mat == -1)).any(), "The adjency mat does not match the search mat"

        zero_point = np.zeros(self.nodes_cnt, dtype=np.int32)
        self.burst_point_index = np.zeros(self.nodes_cnt, dtype=np.int32)
        self.awakening_point_1st_index = np.zeros(self.nodes_cnt, dtype=np.int32)
        self.delta_c_am_milti_s_am = np.zeros(self.nodes_cnt)
        self.ta_tm_indicator = np.zeros(self.nodes_cnt)
        for f_idx in range(self.nodes_cnt):  # FIXME a slow loop here
            l_idx_list = self.search_mat[f_idx].tolist()
            this_tx_list = []
            for l_idx in l_idx_list:
                if l_idx == -1:
                    continue
                this_tx_list.extend(self.search_list[l_idx])
            this_tx_list.sort(key=lambda x: x[1])
            if len(this_tx_list) <= 1:
                continue
            value_list = np.array([i[self.value_index - 2] for i, _ in this_tx_list])
            ts_list = np.array([datetime.datetime.strptime(i[self.datetime_index - 2], self.timeformat).timestamp() for i, _ in this_tx_list])
            self.burst_point_index[f_idx] = np.argmax(value_list)

            c_0, c_m = value_list[zero_point[f_idx]], value_list[self.burst_point_index[f_idx]]
            t_0, t_m = ts_list[zero_point[f_idx]], ts_list[self.burst_point_index[f_idx]]
            if zero_point[f_idx] == self.burst_point_index[f_idx] or c_0 == c_m or t_0 == t_m:
                continue
            self.awakening_point_1st_index[f_idx] = np.argmax(
                np.abs(
                    (c_m - c_0) * ts_list[zero_point[f_idx]:self.burst_point_index[f_idx] + 1] - 
                    (t_m - t_0) * value_list[zero_point[f_idx]:self.burst_point_index[f_idx] + 1] + 
                    t_m * c_0 - c_m * t_0
                    ) / 
                np.sqrt((c_m - c_0) ** 2 + (t_m - t_0) ** 2)
            )

            c_a, t_a = value_list[self.awakening_point_1st_index[f_idx]], ts_list[self.awakening_point_1st_index[f_idx]]
            self.ta_tm_indicator[f_idx] = self.burst_point_index[f_idx] - self.awakening_point_1st_index[f_idx] + 1
            self.delta_c_am_milti_s_am[f_idx] = (c_m - c_a) ** 2 / (t_m - t_a)

    def __build_timeslot_sequence(self):
        data_list = []
        for og_index, row in self.tensor.data.iterrows():
            _, _, *datas = row
            dt = datas[self.datetime_index - 2]
            ts = int(datetime.datetime.strptime(dt, self.timeformat).timestamp())
            data_list.append((datas, ts, og_index))
        data_list.sort(key=lambda x: x[1])
        first_dt = datetime.datetime.fromtimestamp(data_list[0][1])
        base_dt = datetime.datetime(year=first_dt.year, month=first_dt.month, day=first_dt.day)
        self.timeslot_base = int(base_dt.timestamp())
        index_map = []
        for datas, ts, og_index in data_list:
            index = self.__get_timeslot_index(ts)
            while len(self.timeslots) < (index + 1):
                self.timeslots.append(list())
            self.timeslots[index].append(datas)
            self.timeorder_datas.append((datas, ts))
            index_map.append(og_index)
        self.timeorder_datas_index_map = np.array(index_map)
        self.timeorder_datas_reverse_index_map = np.zeros_like(self.timeorder_datas_index_map)
        self.timeorder_datas_reverse_index_map[self.timeorder_datas_index_map] = np.arange(len(self.timeorder_datas_index_map))
        for item in self.timeslots:
            self.timeslot_index.append(len(item))

    def __get_timeslot_index(self, timestamp: Union[str, int]):
        if isinstance(timestamp, str):
            timestamp = int(datetime.datetime.strptime(timestamp, self.timeformat).timestamp())

        base_delta = timestamp - self.timeslot_base
        if base_delta < 0:
            return 0
        return int(base_delta // (self.timeslot * 60 * 60 * 24))

    def update_suspiciousness(self, index: Union[int, Iterable[int]]=-1):
        """Update node(s) suspiciousness.

        Args:
            index (Union[int, Iterable[int]]): specified which node(s) to be updated, -1 indicates whole graph, default -1.
        """
        if isinstance(index, Iterable):
            update_index_list = [i for i in index if i >= 0]
        else:
            if index >= 0:
                update_index_list = [index]
            else:
                update_index_list = list(range(int(max(*self.adj.shape))))
        update_index_list = np.array(update_index_list)  # K = len(update_index_list)
        update_index_list_expand = np.zeros(self.nodes_cnt, dtype=np.bool8)
        update_index_list_expand[update_index_list] = True

        related_nodes = self.adj[:, update_index_list].T
        # topological suspiciousness
        t1 = related_nodes @ self.value_array[0][self.search_mat]
        E_k_i = np.sum(t1, axis=1)
        E_j_i = np.sum(t1[:, (self.subset_indicator == 1)], axis=1)
        E_k_i[E_k_i == 0] = E_j_i[E_k_i == 0]
        alpha = E_j_i / E_k_i

        # temporal suspiciousness
        fi_T_i_V = np.sum((related_nodes * self.delta_c_am_milti_s_am) * ((self.burst_point_index * related_nodes) - (self.awakening_point_1st_index * related_nodes) + 1), axis=1)
        fi_T_i_S = np.sum((related_nodes * self.delta_c_am_milti_s_am) * ((self.burst_point_index * related_nodes) - (self.awakening_point_1st_index * related_nodes) + 1)[:, (self.subset_indicator == 1)], axis=1)
        fi_T_i_V[fi_T_i_V == 0] = fi_T_i_S[fi_T_i_V == 0]
        beta = fi_T_i_S / fi_T_i_V

        # monetary suspiciousness
        ita_S = np.sum((self.adj[:, (self.subset_indicator == 1)]) @ self.value_array[0][self.search_mat])
        ita_V_n_S = np.sum(self.value_array[0][self.search_mat]) - ita_S + 1e-18
        try:
            bal = np.min([ita_S / ita_V_n_S, ita_V_n_S / ita_S])
        except Exception as e:
            print(ita_S, ita_V_n_S)
            raise e
        R_S_list = self.R_list[np.where(self.subset_indicator == 1 & update_index_list_expand)]
        R_V_n_S_list = self.R_list[np.where(self.subset_indicator == 0 & update_index_list_expand)]
        if R_V_n_S_list.shape[0] <= 5:
            gamma = np.ones_like(R_S_list)
        else:
            gamma = bal * stats.entropy(R_S_list, R_V_n_S_list)

        # suspiciousness fusion
        self.suspiciousness[update_index_list] = np.sum((related_nodes @ self.value_array[0][self.search_mat]) * (self.b ** (alpha + beta + gamma - 3)).reshape((-1, 1)), axis=1)

    def __str__(self):
        return str(vars(self))
    
    def run(self):
        ...
    
    def anomaly_detection(self):
        ...
    
    def save(self):
        ...
