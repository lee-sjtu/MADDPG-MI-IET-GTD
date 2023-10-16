import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage1 = []
        # self._storage2 = []
        self._maxsize1 = int(size)
        # self._maxsize2 = int(size*1/4)
        self._next_idx1 = 0
        # self._next_idx2 = 0

    def __len__(self):
        return len(self._storage1)

    def add1(self, obs_t, action, reward, obs_tp1, done):
        data1 = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx1 >= len(self._storage1):
            self._storage1.append(data1)
        else:
            self._storage1[self._next_idx1] = data1
        self._next_idx1 = (self._next_idx1 + 1) % self._maxsize1

    # def add2(self, obs_t, action, reward, obs_tp1, done):
    #     data2 = (obs_t, action, reward, obs_tp1, done)
    #     if self._next_idx2 >= len(self._storage2):
    #         self._storage2.append(data2)
    #     else:
    #         self._storage2[self._next_idx2] = data2
    #     self._next_idx2 = (self._next_idx2 + 1) % self._maxsize2

    def _encode_sample(self, idxes1):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes1:
            data1 = self._storage1[i]
            obs_t1, action1, reward1, obs_tp11, done1 = data1
            obses_t.append(np.array(obs_t1, copy=False))
            actions.append(np.array(action1, copy=False))
            rewards.append(reward1)
            obses_tp1.append(np.array(obs_tp11, copy=False))
            dones.append(done1)
        # for i in idxes2:
        #     data2 = self._storage2[i]
        #     obs_t2, action2, reward2, obs_tp12, done2 = data2
        #     obses_t.append(np.array(obs_t2, copy=False))
        #     actions.append(np.array(action2, copy=False))
        #     rewards.append(reward2)
        #     obses_tp1.append(np.array(obs_tp12, copy=False))
        #     dones.append(done2)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index1(self, batch_size):
        return [random.randint(0, len(self._storage1) - 1) for _ in range(batch_size)] #index_n

    def make_index2(self, batch_size):
        return [random.randint(0, len(self._storage2) - 1) for _ in range(batch_size)] #index_n

    # def make_latest_index(self, batch_size):
    #     idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
    #     np.random.shuffle(idx)
    #     return idx

    def sample_index(self, idxes1):
        return self._encode_sample(idxes1)

    # def sample(self, batch_size):
    #     if batch_size > 0:
    #         idxes1 = self.make_index1(batch_size*0.75)
    #         idxes2 = self.make_index2(batch_size*0.25)
    #     else:
    #         idxes1 = range(0, len(self._storage1))
    #         idxes2 = range(0, len(self._storage2))
    #     return self._encode_sample(idxes1, idxes2)

    # def collect(self):
    #     return self.sample(-1)
