import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *
from model.architecture.bert import BERT
from model.architecture.bert_pretrain import BERTLM
from dataset.dataset import FlowDataset
from dataset.vocab import TimeVocab, SizeVocab
import argparse

class BertTrainer:
    def __init__(self, args, bert: BERT, timevocab: TimeVocab, sizevocab: SizeVocab):
        self.args = args
        self.device = set_device(self.args.trainer.device)
        self.bert = bert
        # Initialize the BERT Language Model, with next_sentence = false, masked lm = True
        self.model = BERTLM(bert, len(timevocab), len(sizevocab)).to(self.device)
        
        if args.trainer.load_save_pth is not None and os.path.exists(args.trainer.load_save_pth):
            self.model.load_state_dict(torch.load(args.trainer.load_save_pth))
            print('{} model load success!'.format(args.trainer.load_save_pth))

        if isinstance(self.args.trainer.device, list) and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=self.args.trainer.device)

        train_dataset = FlowDataset(args, timevocab, sizevocab, train=True)
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.trainer.batch_size, shuffle=True,
                                           drop_last=True)
        print('total len of train dataset: ', len(train_dataset))
        test_dataset = FlowDataset(args, timevocab, sizevocab, train=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.trainer.batch_size, shuffle=False, drop_last=True)

        self.optim = AdamW(self.model.parameters(), lr=self.args.trainer.optimizer.lr,
                                      betas=eval(self.args.trainer.optimizer.beta),
                                      weight_decay=self.args.trainer.optimizer.weight_decay)

        self.criterion = nn.NLLLoss(ignore_index=0)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.best_ipd_acc = 0.0
        self.best_size_acc = 0.0

    def train(self, epoch):
        self.iterator(epoch, self.train_dataloader, is_train=True)

    def test(self, epoch):
        self.iterator(epoch, self.test_dataloader, is_train=False)

    def iterator(self, epoch, dataloader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        data_iter = tqdm(enumerate(dataloader), desc='training' if is_train else 'testing' + ' step: {}'.format(epoch),
                        total=len(dataloader))
        
        total_loss = 0.0
        total_ipd_correct = 0.0
        total_ipd_element = 0.0
        total_size_correct = 0.0
        total_size_element = 0.0
        for iter, batch in data_iter:
            batch = {key: value.to(self.device) for key, value in batch.items()}

            ipd_pred, size_pred = self.model(batch['flow_ipd'], batch['flow_size'])

            ipd_loss = self.criterion(ipd_pred.transpose(1, 2), batch['flow_ipd_label'])
            size_loss = self.criterion(size_pred.transpose(1, 2), batch['flow_size_label'])
            loss = ipd_loss + size_loss

            ipd_label_mask = batch['flow_ipd_label'].ne(0)
            size_label_mask = batch['flow_size_label'].ne(0)

            batch_ipd_correct = (ipd_pred.argmax(dim=-1).eq(batch['flow_ipd_label']) & ipd_label_mask).sum().item()
            batch_total_ipd = ipd_label_mask.sum().item()

            batch_size_correct = (size_pred.argmax(dim=-1).eq(batch['flow_size_label']) & size_label_mask).sum().item()
            batch_total_size = size_label_mask.sum().item()

            total_ipd_correct += batch_ipd_correct
            total_ipd_element += batch_total_ipd
            total_size_correct += batch_size_correct
            total_size_element += batch_total_size

            total_loss += loss.item()

            if is_train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()


            post_fix = {
                'epoch': epoch_iter,
                'iter': iter,
                'avg_loss': total_loss / (iter + 1),
                'ipd_loss': ipd_loss.item(),
                'size_loss': size_loss.item(),
                'batch_ipd_acc': batch_ipd_correct / batch_total_ipd,
                'batch_size_acc': batch_size_correct / batch_total_size, 
            }

            if iter % self.args.trainer.log_frequency == 0:
                data_iter.set_postfix(post_fix)
        
        total_ipd_acc = total_ipd_correct / total_ipd_element
        total_size_acc = total_size_correct / total_size_element

        print('total_ipd_acc: ', total_ipd_acc, end='\t')
        print('total_size_acc: ', total_size_acc)

        if not is_train:
            if total_ipd_acc > self.best_ipd_acc:
                self.best_ipd_acc = total_ipd_acc
                self.save(epoch, best='ipdbest')

            if total_size_acc > self.best_size_acc:
                self.best_size_acc = total_size_acc
                self.save(epoch, best='sizebest')


    def save(self, epoch, best=None):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if best is not None:
            output_path = self.args.trainer.model_save_pth + '_' + best + '.pth'
        else:
            output_path = self.args.trainer.model_save_pth + "_ep%d" % epoch + '.pth'

        torch.save(self.model.cpu().state_dict(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', type=str, default='./trafficMimic/config/bert.yaml',
                        help='path to load config', )
    args = parser.parse_args()
    args = recursive_namespace(read_yaml(args.pth))
   
    # train_data: [time, size], time: list of list
    if not os.path.isfile(args.trainer.timevocab_pth) or not os.path.isfile(args.trainer.sizevocab_pth):
        train_data = list(feature_extract(read_flow_pkl(args.trainer.train_data_pth)))
        print('train data loaded!')
    
    if not os.path.isfile(args.trainer.timevocab_pth):
        timevocab = TimeVocab(list(train_data[0]))
        timevocab.save_vocab(args.trainer.timevocab_pth)
        print('timevocab save success!')
    else:
        timevocab = TimeVocab.load_vocab(args.trainer.timevocab_pth)
        print('timevocab load success!')

    if not os.path.isfile(args.trainer.sizevocab_pth):
        sizevocab = SizeVocab(list(train_data[1]))
        sizevocab.save_vocab(args.trainer.sizevocab_pth)
        print('sizevocab save success!')
    else:
        sizevocab = SizeVocab.load_vocab(args.trainer.sizevocab_pth)
        print('sizevocab load success!')
    
    args.trainer.timevocab = len(timevocab)
    args.trainer.sizevocab = len(sizevocab)
    print(args.trainer.timevocab, args.trainer.sizevocab)
    bert = BERT(args.trainer.timevocab, args.trainer.sizevocab, args.model.hidden, args.model.d_ff,
                args.model.n_layers, args.model.attn_heads, args.model.dropout)
    
    trainer = BertTrainer(args, bert, timevocab, sizevocab)

    print('Training Process Start')
    
    for epoch_iter in range(args.trainer.epochs):
        trainer.train(epoch_iter)
        trainer.save(epoch_iter)

        if trainer.test_dataloader is not None:
            trainer.test(epoch_iter)
