import numpy as np
from collections import defaultdict
from .helper import *

class Dataset(object):

    def __init__(self, filename, use_unpop_weight):
        
        self.use_unpop_weight = use_unpop_weight
        
        self._data = np.load(filename, allow_pickle=True)
        self.train_data = self._data['train_data']
        self.test_data = self._data['test_data'].tolist()
        self._train_index = np.arange(len(self.train_data), dtype=np.uint)
        self._n_users, self._n_items = self.train_data.max(axis=0) + 1

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        self.item_users_list = defaultdict(list)
        
        for u, i in self.train_data:
            self.user_items[u].add(i)
            self.item_users[i].add(u)
            # Get a list version so we do not need to perform type casting

        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        ### Item's max_user
        self._max_user_neighbors = max([len(x) for x in self.item_users.values()])
        ### User's max_item
#         self._max_past_items = max([len(items) for items in self.user_items.values()])
        
        ### Item's Popularity && Un-Popularity score
        self._popularity = [len(x) for x in self.item_users.values()]
        self._item_popularity = defaultdict(dict)
        self.total_users = self.train_data[:, 0].max()+1
        self.item_unpop = defaultdict(dict)
        # for max-min normalization
        self.max_score = 0
        self.min_score = 10
        
        for _, (item, users) in enumerate(self.item_users.items()):
            self._item_popularity[item] = [len(users)]
            unpop_score = np.log(self.total_users / len(users))
            self.item_unpop[item] = unpop_score
            if unpop_score > self.max_score:
                self.max_score = unpop_score
            if unpop_score < self.min_score:
                self.min_score = unpop_score
                
        print('max_unpop_score: {:.3f}\nmin_unpop_score: {:.3f}'.format(self.max_score, self.min_score))
        
        # max-min normalization
        self.item_unpop_normalized = defaultdict(dict)

        for item, score in self.item_unpop.items():
            self.item_unpop_normalized[item] = (score - self.min_score)/(self.max_score-self.min_score)
        
        ### Construct User_unpop_score: [mean, std]
        self.items_score_list = list()
        self.user_unpops = np.zeros((self.total_users, 2))

        for idx, (user, items) in enumerate(self.user_items.items()):
            # current user's items_list (unpop scores)
            self.items_score_list = [self.item_unpop[i] for i in items]
            self.user_unpops[user] = np.mean(self.items_score_list), np.std(self.items_score_list)       
        
        ### Popularity to 2/8 LongTail
        # sort dict with descending order (pop -> not pop)
        self._sorted_pop = sorted(self._item_popularity, key=self._item_popularity.get, reverse=True)
        for item in self._item_popularity.keys():
            if item in self._sorted_pop[:int(0.2*len(self._sorted_pop))]:
                self._item_popularity[item].append(False)  # no Longtail item
            else:
                self._item_popularity[item].append(True)  # Longtail item
        ### the self._item_popularity[item] : 
        ### {0: [23, True], ... } ==>  item: [users_count, longtail_item_or_not]
        
        
    @property
    def train_size(self):
        """
        :return: number of examples in training set
        :rtype: int
        """
        return len(self.train_data)

    @property
    def user_count(self):
        return self._n_users

    @property
    def item_count(self):
        return self._n_items

    def _sample_item(self):
        """
        Draw an item uniformly
        """
        return np.random.randint(0, self.item_count)

    def _sample_negative_item(self, user_id):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        n = self._sample_item()
        positive_items = self.user_items[user_id]

        if len(positive_items) >= self.item_count:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.item_count))
        # keep sampling util the sample is not in Por
        while n in positive_items or n not in self.item_users:
            n = self._sample_item()
        return n

    def _generate_data(self, neg_count):
        idx = 0
        self._examples = np.zeros((self.train_size*neg_count, 3),
                                  dtype=np.uint32)
        self._examples[:, :] = 0
        for user_idx, item_idx in self.train_data:
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                
                # Training Data has 3 cols: [u, item, neg_item]
                self._examples[idx, :] = [user_idx, item_idx, neg_item_idx]
                idx += 1

    def get_data(self, batch_size, neighborhood, neg_count):
        # Allocate inputs
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        # Will use past_users to embed each item
        pos_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)  # item's max_user (who bought it)
        pos_length = np.zeros(batch_size, dtype=np.int32)
        # Will use past_users to embed each neg_item
        neg_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(batch_size, dtype=np.int32)
        # unpop_matching_score
        pos_unpop_mtc, neg_unpop_mtc = np.zeros(batch_size, dtype=np.float32), np.zeros(batch_size, dtype=np.float32)
        
        # Shuffle index
        np.random.shuffle(self._train_index)
        
        # idx: the ith example in a batch
        idx = 0
        for user_idx, item_idx in self.train_data[self._train_index]:
            # TODO: set positive values outside of for loop
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                ### In GMF neighborhood=False
                ### Only find neighbor for CMN
                if neighborhood:
                    if len(self.item_users[item_idx]) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        
#                         print('pos_length[idx]: ', pos_length[idx], '\n', 'item_users_list[item_idx] : ', self.item_users_list[item_idx])
                        
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(self.item_users[neg_item_idx]) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx
                        
                if self.use_unpop_weight:
                    user_unpop = self.user_unpops[user_idx]
                    pos_unpop, neg_unpop = self.item_unpop[item_idx], self.item_unpop[neg_item_idx]
                    pos_unpop_norm, neg_unpop_norm = self.item_unpop_normalized[item_idx], self.item_unpop_normalized[neg_item_idx]

                    pos_unpop_mtc[idx], neg_unpop_mtc[idx] = unpop_matching_score(user_unpop, pos_unpop, neg_unpop,
                                                                                  pos_unpop_norm, neg_unpop_norm)
                    

                idx += 1
                # Yield batch if we filled out a Batch
                if idx == batch_size:
                    
                    if neighborhood and self.use_unpop_weight:
                        max_length = max(neg_length.max(), pos_length.max())
#                         max_past_item = past_items_length.max()
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                              neg_neighbor[:, :max_length], neg_length, \
                              pos_unpop_mtc, neg_unpop_mtc, \
#                               past_items[:, :max_past_item], past_items_length
                              
                    
                    elif not neighborhood and self.use_unpop_weight:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_unpop_mtc, neg_unpop_mtc, \
#                               past_items[:, :max_past_item], past_items_length
                        
                        
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            
            if neighborhood and self.use_unpop_weight:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                      neg_neighbor[:idx, :max_length], neg_length[:idx], \
                      pos_unpop_mtc[:idx], neg_unpop_mtc[:idx], \
#                       past_items[:idx, :max_past_item], past_items_length[:idx]
                
            elif not neighborhood and self.use_unpop_weight:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_unpop_mtc[:idx], neg_unpop_mtc[:idx], \
#                       past_items[:idx, :max_past_item], past_items_length[:idx]
            else:
                yield batch[:idx]
