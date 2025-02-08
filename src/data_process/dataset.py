import os
import copy
import torch
import random
import numpy as np
import collections

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

class KGCDataModule:
    def __init__(self, args: dict):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']
        self.seed = args['seed']

        self.add_ent_neighbors = True if args['add_ent_neighbors'] == 'True' else False
        self.add_rel_neighbors = True if args['add_rel_neighbors'] == 'True' else False
        self.ent_neighbor_num = args['ent_neighbor_num']
        self.rel_qual_neighbor_num = args['rel_qual_neighbor_num']
        self.ent_qual_neighbor_num = args['ent_qual_neighbor_num']
        self.no_entity_token = args['no_entity_token']
        self.no_relation_token = args['no_relation_token']
        self.no_qual_relation_token = args['no_qual_relation_token']
        self.no_qual_entity_token = args['no_qual_entity_token']
        self.dataset_mode = args['dataset_mode']
        self.train_mode = args['train_mode']
        self.device = args['device']

        # 1. read all data lines
        self.lines = self.read_lines()
        # 2。 get the entities and relations mapping information
        self.entities, self.relations = self.read_entities_and_relations()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')
        # 3. construct the vocab dictionary
        self.vocab, self.reverse_vocab, self.vocab_offset = self.get_vocab()
        args.update(self.vocab_offset)
        # 4. get the number of relations and entities
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = self.vocab_offset['relation_end_idx'] - self.vocab_offset['relation_begin_idx']
        args['num_entities'] = self.vocab_offset['entity_end_idx'] - self.vocab_offset['entity_begin_idx']
        args['num_specials'] = self.vocab_offset['entity_begin_idx']
        # 5. get the entities neighbor
        self.neighbors = self.get_triple_neighbors()
        # 6. entities to be filtered when predict some triplet
        self.triple_entity_filter = self.get_triple_entity_filter()
        self.qual_entity_filter = self.get_qualifier_entity_filter()
        self.qual_entity_filter_key_rel = self.get_qualifier_entity_filter_key_relation()
        # 7. create examples
        examples = self.create_examples()
        # 8. the above inputs are used to construct pytorch Dataset objects
        self.train_ds = KGCDataset(examples['train'])
        self.dev_ds = KGCDataset(examples['dev'])
        self.test_ds = KGCDataset(examples['test'])

    def read_lines(self):
        """
        read triplets from files, we need add the reverse data
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    reverse_split_line = []
                    split_line = str(line).strip().split('\t')
                    h, r, t = split_line[0], split_line[1], split_line[2]
                    qual = split_line[3:]
                    reverse_split_line.append(t)
                    reverse_split_line.append(r+'_reverse')
                    reverse_split_line.append(h)
                    for i in range(len(qual) // 2):
                        reverse_split_line.append(qual[i*2]+'_reverse')
                        reverse_split_line.append(qual[i*2+1])
                    raw_data.append(tuple(split_line))
                    raw_data.append(tuple(reverse_split_line))
            lines[mode] = raw_data

        return lines

    def read_entities_and_relations(self):
        """
        read entities and realtions information
        :return:
        """
        entities_list = []
        relations_list = []
        for split_name in self.lines:
            split_data = self.lines[split_name]
            for data in split_data:
                q1, p, q2 = data[0], data[1], data[2]
                if q1 not in entities_list:
                    entities_list.append(q1)
                if p not in relations_list:
                    relations_list.append(p)
                if q2 not in entities_list:
                    entities_list.append(q2)
                if len(data) > 3:
                    data = data[3:]
                    for i in range(len(data)):
                        if i % 2 == 0:
                            if data[i] not in relations_list:
                                relations_list.append(data[i])
                        else:
                            if data[i] not in entities_list:
                                entities_list.append(data[i])
        entities_dict = {}
        relations_dict = {}
        count = 0
        for ent in entities_list:
            entities_dict[ent] = {'token_id': count}
            count = count + 1
        count = 0
        for rel in relations_list:
            relations_dict[rel] = {'token_id': count}
            count = count + 1

        return entities_dict, relations_dict

    def get_vocab(self):
        """
        construct the vocab
        :return: two Python Dict
        """
        tokens = ['[PAD]', '[MASK]', '[SEP]', '[CLS]', self.no_entity_token, self.no_relation_token, self.no_qual_entity_token, self.no_qual_relation_token]
        entity_names = [e for e in self.entities]
        relation_names = [r for r in self.relations]

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        reverse_vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx
            reverse_vocab[idx] = token

        return vocab, reverse_vocab, {
            'entity_begin_idx': entity_begin_idx,
            'entity_end_idx': entity_end_idx,
            'relation_begin_idx': relation_begin_idx,
            'relation_end_idx': relation_end_idx,
        }

    def get_triple_neighbors(self):
        """
        construct neighbor prompts from training dataset, only for triple part
        :return:
        """
        if self.train_mode == 'without_valid':
            lines = self.lines['train']
        elif self.train_mode == 'with_valid':
            lines = self.lines['train'] + self.lines['dev']
        neighbor_data = {e: {'triple_neighbors': [], 'triple_qual_neighbors': []} for e in self.entities}

        triple_lines = []
        for line in lines:
            triple_lines.append(tuple(list(line)[0:3]))

        # 1. get (h, r, t)'s triple neighbor
        for h, r, t in triple_lines:
            tri_neighbors = [self.vocab['[MASK]'], self.vocab[r], self.vocab[t]]
            neighbor_data[h]['triple_neighbors'].append(tri_neighbors)

        # 2. get (h, r, t, qp)'s triple qualifier neighbor
        for line in lines:
            tri_qual_neighbors = [self.vocab['[MASK]'], self.vocab[line[1]], self.vocab[line[2]]]
            qual_line = line[3:]
            pad_len = 2 * self.ent_qual_neighbor_num - len(qual_line)
            for i in range(len(qual_line)):
                tri_qual_neighbors.append(self.vocab[qual_line[i]])
            for i in range(pad_len):
                tri_qual_neighbors.append(self.vocab['[PAD]'])
            if pad_len < 0:
                tri_qual_neighbors = tri_qual_neighbors[0: 3 + 2 * self.ent_qual_neighbor_num]
            neighbor_data[line[0]]['triple_qual_neighbors'].append(tri_qual_neighbors)

        # 3. add a fake neighbor if there is no neighbor for the entity
        for nei in neighbor_data:
            if len(neighbor_data[nei]['triple_neighbors']) == 0:
                tri_neighbors = [self.vocab['[MASK]'], self.vocab[self.no_relation_token], self.vocab[self.no_entity_token]]
                neighbor_data[nei]['triple_neighbors'].append(tri_neighbors)

                tri_qual_neighbors = [self.vocab['[MASK]'], self.vocab[self.no_relation_token], self.vocab[self.no_entity_token]]
                pad_len = 2 * self.ent_qual_neighbor_num
                for i in range(pad_len):
                    tri_qual_neighbors.append(self.vocab['[PAD]'])
                neighbor_data[nei]['triple_qual_neighbors'].append(tri_qual_neighbors)

        return neighbor_data

    def get_triple_entity_filter(self):
        """
        only get the entity filter entity list
        :return:
        """
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        triple_lines = []
        for line in lines:
            triple_lines.append(tuple(list(line)[0:3]))
        for h, r, t in triple_lines:
            entity_filter[self.entities[h]['token_id'], self.relations[r]['token_id']].add(self.entities[t]['token_id'])
        return entity_filter

    def get_qualifier_entity_filter(self):
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        res = collections.defaultdict(lambda: collections.defaultdict(list))
        for triplet in lines:
            triplet_len = len(triplet)
            real_triplet = copy.deepcopy(triplet[:triplet_len])
            re_pair = [self.entities[real_triplet[0]]['token_id'], self.relations[real_triplet[1]]['token_id'], self.entities[real_triplet[2]]['token_id']]
            for q, v in zip(real_triplet[3::2], real_triplet[4::2]):
                re_pair.append(self.relations[q]['token_id'])
                re_pair.append(self.entities[v]['token_id'])
            qv_num = len(triplet[3:]) // 2
            pos_list = [2]
            for num in range(qv_num):
                pos_list.append(3+2*num+1)
            for pos in pos_list:
                key = " ".join([
                    str(re_pair[x]) for x in range(len(re_pair)) if x != pos
                ])
                res[pos][key].append(re_pair[pos])

        return res

    def get_qualifier_entity_filter_key_relation(self):
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)
        for line in lines:
            tri_qual_list = line[3:]
            tri_qual_num = len(tri_qual_list) // 2
            for i in range(tri_qual_num):
                entity_filter[self.relations[tri_qual_list[2*i]]['token_id']].add(self.entities[tri_qual_list[2*i+1]]['token_id'])
        return entity_filter

    def create_examples(self):
        """
        create all examples
        :return:
        """
        examples = dict()
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            triple_lines = []
            for line in lines:
                triple_lines.append(tuple(list(line)[0:3]))
            count = 0
            for h, r, t in tqdm(triple_lines, desc=f'[{mode}]create examples'):
                example = self.create_main_triple_example(lines[count], h, r, t)

                data.append(example)
                count = count + 1
            examples[mode] = data
        return examples

    def create_main_triple_example(self, tri_qual, h, r, t):
        """
        create main triple example, only mask the head or tail entity
        """
        mask_token = '[MASK]'
        pad_token = '[PAD]'

        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]
        # 1. get entity filters
        tail_filters = list(self.triple_entity_filter[head['token_id'], rel['token_id']] - {tail['token_id']})
        # 2. get entity and qualifier filter
        triplet_len = len(tri_qual)
        real_triplet = copy.deepcopy(tri_qual[:triplet_len])
        re_pair = [self.entities[real_triplet[0]]['token_id'], self.relations[real_triplet[1]]['token_id'], self.entities[real_triplet[2]]['token_id']]
        for q, v in zip(real_triplet[3::2], real_triplet[4::2]):
            re_pair.append(self.relations[q]['token_id'])
            re_pair.append(self.entities[v]['token_id'])
        mask_pos = 2
        # 3. get qualifier pairs and qualifier sequence
        tri_qual_list = tri_qual[3:]
        tri_qual_num = len(tri_qual_list) // 2
        # 3.1 if the qualifier number larger than the setting relation qualifier neighbor length, then we need to sample
        if tri_qual_num > self.rel_qual_neighbor_num:
            filter_tri_qual_list = []
            random_list = []
            for i in range(tri_qual_num):
                random_list.append([tri_qual_list[2 * i], tri_qual_list[2 * i + 1]])
            for i in range(self.rel_qual_neighbor_num):
                filter_tri_qual_list.append(random_list[i][0])
                filter_tri_qual_list.append(random_list[i][1])
            tri_qual_list = filter_tri_qual_list
        # 3.2 get the triple qualifier pairs id list information
        tri_qual_prompt = []
        for i in range(len(tri_qual_list) // 2):
            tri_qual_prompt.append([self.vocab[tri_qual_list[2 * i]], self.vocab[tri_qual_list[2 * i + 1]], self.vocab[mask_token]])
        # 3.3 if the qualifier pairs number is none, then we need add one qualifier pair with (no_qual_relation_token, no_qual_entity_token)
        pad_len = self.rel_qual_neighbor_num - len(tri_qual_prompt)
        if len(tri_qual_prompt) == 0:
            tri_qual_prompt.append([self.vocab[self.no_qual_relation_token], self.vocab[self.no_qual_entity_token], self.vocab[mask_token]])
        # 3.4 change the qualifier list to qualifier sequence
        tri_qual_sequence = []
        tri_qual_sequence.append(self.vocab[h])
        tri_qual_sequence.append(self.vocab[r])
        tri_qual_sequence.append(self.vocab[mask_token])
        for i in range(len(tri_qual_list) // 2):
            tri_qual_sequence.append(self.vocab[tri_qual_list[2 * i]])
            tri_qual_sequence.append(self.vocab[tri_qual_list[2 * i + 1]])
        # 3.5 if the qualifier pairs smaller than the relation qualifier neighbor length, then we need to pad
        for i in range(pad_len):
            tri_qual_sequence.append(self.vocab[pad_token])
            tri_qual_sequence.append(self.vocab[pad_token])
        # 4. prepare examples
        example = {
            'tri':  (h, r, t),
            'tri_seq':  [self.vocab[h], self.vocab[r], self.vocab[mask_token]],
            'tri_qual_seq': tri_qual_sequence,
            'rel_qual_neighbors': tri_qual_prompt,
            'label': tail["token_id"],
            'neighbor_label': head["token_id"],
            'entity_filters': tail_filters,
            'qual_filters': tail_filters,
            'mask_pos': 2
        }
        return example

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        data_triple = [data_dit['tri'] for data_dit in batch_data]
        # 1. get entity neighbors from KG
        if self.add_ent_neighbors:
            batch_ent_neighbors = [[] for _ in range(self.ent_neighbor_num)]
            batch_ent_neighbors_with_qual = [[] for _ in range(self.ent_neighbor_num)]
            for ent, _, _ in data_triple:
                ent_neighbors = self.neighbors[ent]['triple_neighbors']
                idxs = list(range(len(ent_neighbors)))
                # 1.1 if the entity's neighbor is greater than ent_neighbor_num, then cut out, else sample if it is less than ent_neighbor_num
                if len(idxs) >= self.ent_neighbor_num:
                    idxs = random.sample(idxs, self.ent_neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.ent_neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.ent_neighbor_num
                for i, idx in enumerate(idxs):
                    batch_ent_neighbors[i].append(ent_neighbors[idx])
                # 1.2 get the statement corresponding to the specified id
                ent_qual_neighbors = self.neighbors[ent]['triple_qual_neighbors']
                for i, idx in enumerate(idxs):
                    batch_ent_neighbors_with_qual[i].append(ent_qual_neighbors[idx])
            # 1.3 get the batch samples
            ent_neighbors = [self.struc_batch_encoding(batch_ent_neighbors[i]) for i in range(self.ent_neighbor_num)]
            ent_qual_neighbors = [self.struc_batch_encoding(batch_ent_neighbors_with_qual[i]) for i in range(self.ent_neighbor_num)]
        else:
            ent_neighbors = None
            ent_qual_neighbors = None

        # 2. get the relation qualifier neighbors
        rel_qual_neighbors = [data_dit['rel_qual_neighbors'] for data_dit in batch_data]
        batch_rel_qual_neighbors = [[] for _ in range(self.rel_qual_neighbor_num)]
        for single_rel_qual_neighbors in rel_qual_neighbors:
            idxs = list(range(len(single_rel_qual_neighbors)))
            # 2.1 if the relation's neighbor is greater than rel_qual_neighbor_num, then cut out, else sample if it is less than rel_qual_neighbor_num
            if len(idxs) >= self.rel_qual_neighbor_num:
                idxs = random.sample(idxs, self.rel_qual_neighbor_num)
            else:
                tmp_idxs = []
                for _ in range(self.rel_qual_neighbor_num - len(idxs)):
                    tmp_idxs.append(random.sample(idxs, 1)[0])
                idxs = tmp_idxs + idxs
            assert len(idxs) == self.rel_qual_neighbor_num
            for i, idx in enumerate(idxs):
                batch_rel_qual_neighbors[i].append(single_rel_qual_neighbors[idx])
        # 2.2 get the batch samples
        rel_qual_neighbors = [self.struc_batch_encoding(batch_rel_qual_neighbors[i]) for i in range(self.rel_qual_neighbor_num)]
        # 3. get other batch data
        tri_seq = [copy.deepcopy(data_dit['tri_seq']) for data_dit in batch_data]
        tri_seq = self.struc_batch_encoding(tri_seq)
        tri_qual_seq = [copy.deepcopy(data_dit['tri_qual_seq']) for data_dit in batch_data]
        tri_qual_seq = self.struc_batch_encoding(tri_qual_seq)
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        neighbor_labels = torch.tensor([data_dit['neighbor_label'] for data_dit in batch_data])
        mask_pos = torch.tensor([data_dit['mask_pos'] for data_dit in batch_data])
        entity_filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['entity_filters']])
        qual_filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['qual_filters']])

        return {
            'data': data_triple, 'tri_seq': tri_seq, 'tri_qual_seq': tri_qual_seq,
            'ent_neighbors': ent_neighbors, 'ent_qual_neighbors': ent_qual_neighbors, 'rel_qual_neighbors': rel_qual_neighbors,
            'labels': labels, 'neighbor_labels': neighbor_labels, 'mask_pos': mask_pos,
            'entity_filters': entity_filters, 'qual_filters': qual_filters
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn, num_workers=self.num_workers, worker_init_fn=np.random.seed(self.seed),
                                batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_train_dev_dataloader(self):
        dataloader = DataLoader(self.train_ds + self.dev_ds, collate_fn=self.collate_fn, num_workers=self.num_workers, worker_init_fn=np.random.seed(self.seed),
                                batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn, num_workers=self.num_workers, worker_init_fn=np.random.seed(self.seed),
                                batch_size=2 * self.batch_size, pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn, num_workers=self.num_workers, worker_init_fn=np.random.seed(self.seed),
                                batch_size=2 * self.batch_size, pin_memory=self.pin_memory, shuffle=False)
        return dataloader