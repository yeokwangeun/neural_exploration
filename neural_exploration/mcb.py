import numpy as np
import itertools
import random
import torch
import pickle
import os
from tqdm import tqdm
from sklearn.decomposition import PCA


class MultimodalContextualBandit:
    def __init__(self,
                 T,
                 num_items,
                 data_path,
                 simulator_path,
                 seed=None,
                 num_pca=0,
                 topk=100
                 ):
        # if not None, freeze seed for reproducibility
        self._seed(seed)

        # number of rounds
        self.T = T
        # simulator path
        self.sim_path = simulator_path
        # item feats
        self.n_arms = num_items
        self.num_pca = num_pca
        self.item_feats = self.get_item_feats(data_path)
        self.item_feats = self.reduce_dim(self.item_feats)
        self.n_item_features = self.item_feats.shape[1]
        self.n_features = self.n_item_features * 2 # user item concat
        # topk
        self.topk = topk

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reduce_dim(self, item_feats):
        if self.num_pca:
            pca = PCA(n_components=self.num_pca)
            pca.fit(item_feats)
            item_feats = pca.transform(item_feats)
        else:
            ##### TODO: dimension reduction #####
            pass
        return item_feats

    def get_item_feats(self, data_path, pca=False):
        item_feat_path = os.path.join(data_path, "ml-25m_item.pkl")
        with open(item_feat_path, "rb") as pf:
            item_feats = pickle.load(pf)
        # summation of image and text
        return np.vstack([
            item_feats[0][i] + item_feats[1][i] 
            for i in range(1, self.n_arms + 1)
        ])

    def reset(self, user_id):
        """Generate new features and new rewards.
        """
        with open(os.path.join(self.sim_path, f"{user_id}.pkl"), "rb") as pf:
            sim = pickle.load(pf)
        seq = sim["seq"][:self.T]
        self.rewards = np.zeros((len(seq), self.n_arms)) # T, n_arms
        self.features = np.zeros((len(seq), self.n_arms, self.n_features)) # T, n_arms, n_features
        user_emb = np.zeros(self.n_item_features)
        
        for i, item in enumerate(tqdm(seq)):
            ##### TODO: context feature generation #####
            user_emb += self.item_feats[item - 1]
            user_emb_repeat = np.repeat(user_emb.reshape(1, -1), self.n_arms, axis=0)
            self.features[i] = np.concatenate([user_emb_repeat, self.item_feats], axis=1)
            score = sim["score"][i].detach().cpu().numpy()
            threshold = score[(-score).argsort()[self.topk]]
            self.rewards[i] = np.where(score > threshold, 1, 0)
            
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
