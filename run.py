import os
import json
import torch
import shutil
import random
import argparse
import numpy as np
import torch.multiprocessing
import datetime

from time import time
from src import KGCDataModule
from src import HyperMono, Knowformer
from src import save_model, score2str

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.multiprocessing.set_sharing_strategy('file_system')

def get_args():
    parser = argparse.ArgumentParser()
    # 1. about training
    parser.add_argument('--task', type=str, default='train', help='train | validate')
    parser.add_argument('--model_path', type=str, default='', help='load saved model for validate')
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='select a gpu like cuda:0')
    parser.add_argument('--dataset', type=str, default='jf17k_100', help='select a dataset')
    # 2. about neighbors
    parser.add_argument('--add_ent_neighbors', type=str, default='True', help='whether use entity neighbor')
    parser.add_argument('--add_rel_neighbors', type=str, default='True', help='whether use relation neighbor')
    parser.add_argument('--ent_neighbor_num', type=int, default=3, help='the entity neighbor numbers')
    parser.add_argument('--rel_qual_neighbor_num', type=int, default=2, help='the relation qual neighbor numbers')
    parser.add_argument('--ent_qual_neighbor_num', type=int, default=2, help='the entity qual neighbor numbers, mainly for get entity embedding')
    parser.add_argument('--no_entity_token', type=str, default='[E_None]', help='you did not set')
    parser.add_argument('--no_relation_token', type=str, default='[R_None]', help='you did not set')
    parser.add_argument('--no_qual_entity_token', type=str, default='[QE_None]', help='you did not set')
    parser.add_argument('--no_qual_relation_token', type=str, default='[QR_None]', help='you did not set')
    # 3. about triple encoder
    parser.add_argument('--kge_lr', type=float, default=6e-4)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--input_dropout_prob', type=float, default=0.7)
    parser.add_argument('--context_dropout_prob', type=float, default=0.1)
    parser.add_argument('--qual_dropout_prob', type=float, default=0.3)
    parser.add_argument('--attention_dropout_prob', type=float, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--entity_dropout_prob', type=float, default=0.1)
    parser.add_argument('--residual_dropout_prob', type=float, default=0.)
    parser.add_argument('--intermediate_size', type=int, default=2048)
    # 5. some unimportant parameters, only need to change when your server/pc changes, I do not change these
    parser.add_argument('--num_workers', type=int, default=32, help='num workers for Dataloader')
    parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    # 6. other setting
    parser.add_argument('--dataset_mode', type=str, default='statement', choices=['triple', 'statement'], help='the dataset mode')
    parser.add_argument('--train_mode', type=str, default='without_valid', choices=['without_valid', 'with_valid'], help='the train mode')
    parser.add_argument('--save_dir_name', type=str, default='default')
    parser.add_argument('--seed', type=int, default=2023, help='set seed')

    # 7. ablation parameters
    parser.add_argument('--beta_weight', type=float, default=4.0)
    parser.add_argument('--hidden_size', type=int, default=400)
    parser.add_argument('--kge_label_smoothing', type=float, default=0.8)
    parser.add_argument('--ent_neighbor_loss_weight', type=float, default=1.0)
    parser.add_argument('--initializer_range', type=float, default=0.02)
    parser.add_argument('--ablation_mode', type=str, default='normal', choices=['normal', 'local', 'global'], help='the ablation mode')

    args = parser.parse_args()
    args = vars(args)
    print(args)

    root_path = os.path.dirname(__file__)
    # 1. saved model_path
    if args['task'] == 'validate':
        args['model_path'] = os.path.join(root_path, args['model_path'])
    # 2. data path
    args['data_path'] = os.path.join(root_path, 'dataset', args['dataset'])
    # 3. output path
    output_dir = os.path.join(root_path, 'output', args['dataset'], 'model', args['train_mode'] + '_' + args['save_dir_name'])
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    args['output_path'] = output_dir

    # 4. save hyper params
    with open(os.path.join(args['output_path'], 'args.txt'), 'w') as f:
        json.dump(args, f, indent=4, ensure_ascii=False)

    # set random seed
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    train_tri_qual_datasets_list = ['wd50k', 'wikipeople', 'jf17k', 'wikipeople-3', 'wikipeople-4', 'jf17k-3', 'jf17k-4']
    valid_tri_qual_datasets_list = ['wd50k', 'wikipeople', 'jf17k', 'wikipeople-3', 'wikipeople-4', 'jf17k-3', 'jf17k-4']
    extra_encoder_datasets_list = ['wd50k_33', 'wd50k_66', 'wd50k_100', 'wikipeople_33', 'wikipeople_66', 'wikipeople_100', 'jf17k_33', 'jf17k_66', 'jf17k_100']
    if args['dataset'] in train_tri_qual_datasets_list:
        args['loss_train_ablation'] = 'tri_qual'
    else:
        args['loss_train_ablation'] = 'qual'
    if args['dataset'] in valid_tri_qual_datasets_list:
        args['loss_valid_ablation'] = 'tri_qual'
    else:
        args['loss_valid_ablation'] = 'qual'
    if args['dataset'] in extra_encoder_datasets_list:
        args['extra_encoder'] = True
    else:
        args['extra_encoder'] = False
    return args

def get_model_config(config):
    model_config = dict()

    model_config["device"] = config["device"]
    model_config["dataset_mode"] = config['dataset_mode']
    model_config["train_mode"] = config['train_mode']

    model_config["add_ent_neighbors"] = config['add_ent_neighbors']
    model_config["add_rel_neighbors"] = config['add_rel_neighbors']
    model_config["ent_neighbor_num"] = config['ent_neighbor_num']

    model_config["num_hidden_layers"] = config['num_hidden_layers']
    model_config["num_attention_heads"] = config['num_attention_heads']
    model_config["input_dropout_prob"] = config['input_dropout_prob']
    model_config["context_dropout_prob"] = config['context_dropout_prob']
    model_config["qual_dropout_prob"] = config['qual_dropout_prob']
    model_config["attention_dropout_prob"] = config['attention_dropout_prob']
    model_config["hidden_dropout_prob"] = config['hidden_dropout_prob']
    model_config["entity_dropout_prob"] = config['entity_dropout_prob']
    model_config["residual_dropout_prob"] = config['residual_dropout_prob']
    model_config["intermediate_size"] = config['intermediate_size']

    model_config["vocab_size"] = config['vocab_size']
    model_config["num_entities"] = config['num_entities']
    model_config["num_relations"] = config['num_relations']
    model_config["num_specials"] = config['num_specials']

    model_config["entity_begin_idx"] = config['entity_begin_idx']
    model_config["entity_end_idx"] = config['entity_end_idx']
    model_config["relation_begin_idx"] = config['relation_begin_idx']
    model_config["relation_end_idx"] = config['relation_end_idx']
    model_config["special_begin_idx"] = 0
    model_config["special_end_idx"] = config['num_specials']

    model_config["beta_weight"] = config["beta_weight"]
    model_config["hidden_size"] = config['hidden_size']
    model_config["ablation_mode"] = config['ablation_mode']
    model_config["initializer_range"] = config['initializer_range']

    moe_datasets_list = ['wd50k_33', 'wd50k_66', 'wd50k_100', 'wikipeople_33', 'wikipeople_66', 'wikipeople_100', 'jf17k_33', 'jf17k_66', 'jf17k_100']
    if config['dataset'] in moe_datasets_list:
        model_config["trm_moe"] = True
    else:
        model_config["trm_moe"] = False
    return model_config

def load_model(model_path: str, device: str, kg, edge_type, edge_norm):
    print(f'Loading Model from {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    model_config = state_dict['config']
    model = Knowformer(model_config, kg, edge_type, edge_norm)
    model.load_state_dict(state_dict['model'])
    return model_config, model

class HyperMonoTrainer:
    def __init__(self, config: dict):
        self.is_validate = True if config['task'] == 'validate' else False
        self.output_path = config['output_path']
        self.epoch = config['epoch']
        self.device = config['device']

        self.train_dl, self.dev_dl, self.test_dl, self.num_batches = self._load_dataset(config)
        self.model_config, self.model = self._load_model(config)

        optimizers = self.model.configure_optimizers(config['epoch'] * len(self.train_dl))
        self.opt, self.scheduler = optimizers['optimizer'], optimizers['scheduler']

        self.log_path = os.path.join(self.output_path, 'log.txt')
        with open(self.log_path, 'w') as f:
            pass

    def _load_dataset(self, config: dict):
        data_module = KGCDataModule(config)
        if config['train_mode'] == 'without_valid':
            train_dl = data_module.get_train_dataloader()
        elif config['train_mode'] == 'with_valid':
            train_dl = data_module.get_train_dev_dataloader()
        num_batches = len(train_dl)
        dev_dl = data_module.get_dev_dataloader()
        test_dl = data_module.get_test_dataloader()

        return train_dl, dev_dl, test_dl, num_batches

    def _load_model(self, config):
        if self.is_validate:
            model_config, encoder = load_model(config['model_path'], config['device'])
            model = HyperMono(config, encoder).to(config['device'])
            return model_config, model
        else:
            model_config = get_model_config(config)
            encoder = Knowformer(model_config)
            model = HyperMono(config, encoder).to(config['device'])
            return model_config, model

    def _train_one_epoch(self):
        self.model.train()
        outputs = list()
        for batch_idx, batch_data in enumerate(self.train_dl):
            batch_loss = self.model.training_step(batch_data)
            outputs.append(batch_loss)

            self.opt.zero_grad()
            batch_loss.backward()
            self.opt.step()

        if self.scheduler is not None:
            self.scheduler.step()
        loss = self.model.training_epoch_end(outputs)
        return loss

    def _validate_one_epoch(self, data_loader):
        self.model.eval()
        outputs = list()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                output = self.model.validation_step(batch_data)
                outputs.append(output)

        return self.model.validation_epoch_end(outputs)

    def train(self):
        best_score = None
        best_test_score = None
        for i in range(1, self.epoch + 1):
            # 1. train and validate one epoch
            begin_time = time()
            train_loss = self._train_one_epoch()
            dev_score = self._validate_one_epoch(self.dev_dl)
            test_score = self._validate_one_epoch(self.test_dl)

            # 2. save log of this epoch
            date = datetime.datetime.now()
            date = date.strftime('%Y-%m-%d %H:%M:%S\t')
            log = date + f'[train] epoch: {i}, loss: {train_loss}' + '\n'
            log += date + f'[dev]   epoch: {i}, ' + score2str(dev_score) + '\n'
            log += date + f'[test]  epoch: {i}, ' + score2str(test_score) + '\n'
            log += '=' * 30 + f' {round(time() - begin_time)}s ' + '=' * 30 + '\n'
            print(log)
            with open(self.log_path, 'a') as f:
                f.write(log + '\n')

            # 3. update the best scores, save the best model
            if best_score is None or best_score['MRR'] < test_score['MRR']:
                best_test_score = test_score
                best_score = test_score
                best_score['epoch'] = i
                save_model(self.model_config, self.model.bert_encoder, os.path.join(self.output_path, 'model.bin'))

        # 4. save the log of best epoch, after training
        date = datetime.datetime.now()
        date = date.strftime('%Y-%m-%d %H:%M:%S\t')
        log = date + f'[best test]  epoch: {best_score["epoch"]}, ' + score2str(best_test_score) + '\n'
        print(log)
        with open(self.log_path, 'a') as f:
            f.write(log + '\n')

    def validate(self):
        shutil.rmtree(self.output_path)

        dev_scores = self._validate_one_epoch(self.dev_dl)
        test_scores = self._validate_one_epoch(self.test_dl)
        print(f'[dev] {score2str(dev_scores)}')
        print(f'[test] {score2str(test_scores)}')

    def main(self):
        if self.is_validate:
            self.validate()
        else:
            self.train()

if __name__ == '__main__':
    config = get_args()
    trainer = HyperMonoTrainer(config)
    trainer.main()