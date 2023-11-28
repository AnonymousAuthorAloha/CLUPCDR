import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = "1"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from my_model import Emcdr,LookupEmbedding,Encoder,Augment
from simclr import SimCLR

class Run():
    def __init__(self,config,temperature):
        self.use_cuda = config['use_cuda']
        self.base_model = config['base_model']
        self.root = config['root']
        self.root_model="./model/"
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src
        self.device=torch.device("cuda:0")

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim'] 
        self.layers_num=8        
        self.lr = config['lr']
        self.wd = config['wd']
        self.lr_pre=0.0003
        self.wd_pre=1e-4
        self.temperature=temperature

        self.input_root = self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'

        # self.input_root_model=self.root + 'ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
        #     '/tgt_' + self.tgt + '_src_' + self.src
        # self.model_src_path=self.input_root_model+'./model_src'
        # self.model_tgt_path=self.input_root_model+'./model_tgt'
        # self.model_pre_part_path=self.input_root_model+'./model_pre_part'
        # self.model_pre_part_2_path=self.input_root_model+'./model_pre_part_2'

        self.model_src_path='./model_src'
        self.model_tgt_path='./model_tgt'
        self.model_pre_part_path='./model_pre_part'
        # self.model_pre_part_2_path='./model_pre_part_2'

        self.results = {'emcdr_mae': 10, 'emcdr_rmse': 10}

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long).to(self.device)
            y = torch.tensor(data[y_col].values, dtype=torch.long).to(self.device)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True,drop_last=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data.columns = cols
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20, padding='post')
            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1).to(self.device)
            y = torch.tensor(data[y_col].values, dtype=torch.long).to(self.device)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True,drop_last=True)
            return data_iter
        
    def get_data(self):
        print('========Reading data_src and data_tgt========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        # data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        # print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        return data_src, data_tgt
    
#得到训练数据
    def get_my_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long).to(self.device)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long).to(self.device)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_test, shuffle=True,drop_last=True)
        return data_iter
        
    def get_my_train_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']
        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        x_col = ['uid', 'iid']
        y_col = ['y']
        X = torch.tensor(data[x_col].values, dtype=torch.long).to(self.device)
        y = torch.tensor(data[y_col].values, dtype=torch.long).to(self.device)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_test, shuffle=True,drop_last=True)
        return data_iter
    
    #测试数据集的行：用户id 物品在目标域Id 用户在目标域对此物品的评分 用户在源域的交互序列
    def get_my_test_data(self):
        print('========Reading test data========')
        data = pd.read_csv(self.test_path, header=None)
        cols = ['uid', 'iid', 'y', 'pos_seq']
        x_col = ['uid', 'iid']
        y_col = ['y']
        data.columns = cols
        X = torch.tensor(data[x_col].values, dtype=torch.long).to(self.device)
        y = torch.tensor(data[y_col].values, dtype=torch.long).to(self.device)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_test, shuffle=True,drop_last=True)
        return data_iter

    def train(self, data_loader, model, criterion_1, optimizer):
        model.train()
        for epoch in range(self.epoch):
            print('Pre_Training Epoch {}:'.format(epoch + 1))
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                X.to(self.device)
                y.to(self.device)
                emb = model(X)
                pred= torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                loss = criterion_1(pred, y.squeeze().float())
                model.zero_grad()
                loss.backward()
                optimizer.step()

    def my_train(self, data_loader, model, criterion_1, optimizer, epoch,model_src,model_tgt,model_pre):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for x_batch, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            src_emb = model_src.uid_embedding(x_batch[:,0].unsqueeze(1)).squeeze().to(self.device)
            src_emb=model_pre(src_emb)
            # tgt_emb = model_tgt.uid_embedding(uid.unsqueeze(1)).squeeze().to(self.device)

            uid_src_emb = model(src_emb).to(self.device)
            iid_tgt_emb= model_tgt.iid_embedding(x_batch[:,1].unsqueeze(1)).squeeze().to(self.device)
            # emb=torch.cat((uid_src_emb,iid_tgt_emb),1)
            pred=torch.sum(uid_src_emb * iid_tgt_emb, dim=1).to(self.device)
            loss = criterion_1(pred, y.squeeze().float().to(self.device))
            # print("loss:"+str(loss.item()))
            model.zero_grad()
            loss.backward()
            optimizer.step()

    def my_train_part(self, data_loader, model, criterion_1, optimizer, epoch,model_src,model_tgt):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        for x_batch, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            x_batch=x_batch.to(self.device)
            model_src=model_src.to(self.device)
            model_tgt=model_tgt.to(self.device)
            src_emb = model_src.uid_embedding(x_batch[:,0].unsqueeze(1)).squeeze().to(self.device)
            # src_emb=model_pre(src_emb)
            # tgt_emb = model_tgt.uid_embedding(uid.unsqueeze(1)).squeeze().to(self.device)

            uid_src_emb = model(src_emb).to(self.device)
            iid_tgt_emb= model_tgt.iid_embedding(x_batch[:,1].unsqueeze(1)).squeeze().to(self.device)
            # emb=torch.cat((uid_src_emb,iid_tgt_emb),1)
            pred=torch.sum(uid_src_emb * iid_tgt_emb, dim=1).to(self.device)
            loss = criterion_1(pred, y.squeeze().float().to(self.device))
            # print("loss:"+str(loss.item()))
            model.zero_grad()
            loss.backward()
            optimizer.step()

#  my_train_part_2(train_loader_pre_part,model_pre_part_2,criterion_1,optimizer_pre_part_2,i,model_src,model_tgt,model_pre_part)
    # def my_train_part_2(self, data_loader, model, criterion_1, optimizer, epoch,model_src,model_tgt,model_pre_part):
    #     print('Training Epoch {}:'.format(epoch + 1))
    #     model.train()
    #     epoch_pre=epoch
    #     for x_batch, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
    #         x_batch=x_batch.to(self.device)
    #         y=y.to(self.device)
    #         src_emb = model_src.uid_embedding(x_batch[:,0].unsqueeze(1)).squeeze().to(self.device)
    #         # src_emb=model_pre(src_emb)
    #         tgt_emb = model_tgt.uid_embedding(x_batch[:,0].unsqueeze(1)).squeeze().to(self.device)

    #         tgt_emb_pred=model_pre_part(src_emb)
    #         tgt_emb_final=model(tgt_emb_pred,tgt_emb)
    #         iid_tgt_emb= model_tgt.iid_embedding(x_batch[:,1].unsqueeze(1)).squeeze().to(self.device)
    #         # emb=torch.cat((uid_src_emb,iid_tgt_emb),1)
    #         pred=torch.sum(tgt_emb_final * iid_tgt_emb, dim=1).to(self.device)
    #         loss = criterion_1(pred, y.squeeze().float().to(self.device))
    #         # print("loss:"+str(loss.item()))
    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    def eval_mae(self, model, data_loader, model_src,model_tgt,model_pre):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for x, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):      
                uid_emb= model.mapping.forward(model_pre(model_src.uid_embedding(x[:,0].unsqueeze(1)).squeeze().to(self.device))).to(self.device)
                # uid_emb= model.mapping.forward(model_src.uid_embedding(x[:,0].unsqueeze(1)).squeeze().to(self.device)).to(self.device)
                emb = model_tgt.forward(x)
                emb[:,0,:]=uid_emb
                pred = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)
        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
    
    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse
           
    def main(self):
        print("Method:only_user_embedding_method")
        # print("SimCLR:push together encoded user embeddings in source domain and encoded user embeddings in target domain")
        print("EMCDR:push together encoded user embeddings in source domain and no-encoded user embeddings in target domain")
        print("EMCDR:using ground truth ratting to be project method,only target domain ground ratting")
        print("please show good ends")
        # print("test:only emcdr")
#------------------------------pre train src and tgt-------------------------------------------
        data_src, data_tgt = self.get_data()

        criterion_1 = torch.nn.MSELoss().to(self.device)
        if os.path.exists(self.model_src_path):
            model_src=torch.load(self.model_src_path)
        else:
            model_src=LookupEmbedding(self.uid_all,self.iid_all,self.emb_dim)
            model_src = model_src.cuda()

            optimizer_src = torch.optim.Adam(params=model_src.parameters(), lr=self.lr, weight_decay=self.wd)

            print('=====CDR Pretraining source domain=====')
            self.train(data_src,model_src,criterion_1,optimizer_src)

            #torch.save(model_src,self.model_src_path)

        if os.path.exists(self.model_tgt_path):
            model_tgt=torch.load(self.model_tgt_path)
        else:
            model_tgt=LookupEmbedding(self.uid_all,self.iid_all,self.emb_dim)
            model_tgt = model_tgt.cuda() 

            optimizer_tgt = torch.optim.Adam(params=model_tgt.parameters(), lr=self.lr, weight_decay=self.wd)
                
            print('=====CDR Pretraining target domain=====')
            self.train(data_tgt,model_tgt,criterion_1,optimizer_tgt)

            # torch.save(model_tgt,self.model_tgt_path)

#------------------------------pre train src and tgt-------------------------------------------
        model_src.eval()
        model_tgt.eval()
#-----------------------------pre train------------------------------------------
        model_pre=Encoder(self.emb_dim,self.layers_num)
        model_pre = model_pre.cuda()
        # model_pre_part=Emcdr(self.emb_dim)
        # model_pre_part=model_pre_part.cuda() if self.use_cuda else model_pre_part

        # model_pre_part_2=Attention(vector_dim=self.emb_dim,num_heads=4,num_layers=1,feedforward_dim=1024)
        # model_pre_part_2=model_pre_part_2.cuda() if self.use_cuda else model_pre_part_2        


        # optimizer_pre_part=torch.optim.Adam(params=model_pre_part.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimizer_pre_part_2=torch.optim.Adam(params=model_pre_part_2.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_pre = torch.optim.Adam(params=model_pre.parameters(), lr=self.lr_pre, weight_decay=self.wd_pre)
        
#-----------------------------pre train_part------------------------------------------        
        # print("model_pre_part starting.......")
        # with torch.cuda.device("cuda:0"):
        #     for i in range(20):
        #         self.my_train_part(train_loader_pre_part,model_pre_part,criterion_1,optimizer_pre_part,i,model_src,model_tgt)
        # model_pre_part.eval()

        if os.path.exists(self.model_pre_part_path):
            model_pre_part=torch.load(self.model_pre_part_path)
        else:
            train_loader_pre_part=self.get_my_train_data()

            model_pre_part=Augment(self.emb_dim)
            model_pre_part=model_pre_part.cuda()

            optimizer_pre_part=torch.optim.Adam(params=model_pre_part.parameters(), lr=self.lr, weight_decay=self.wd)
                
            print("model_pre_part starting.......")
            with torch.cuda.device("cuda:0"):
                 for i in range(30):
                     self.my_train_part(train_loader_pre_part,model_pre_part,criterion_1,optimizer_pre_part,i,model_src,model_tgt)

            # torch.save(model_pre_part,self.model_pre_part_path)
        model_pre_part.eval()
#-----------------------------pre train_part------------------------------------------   

#-----------------------------pre train_part_2------------------------------------------   

        # train_loader_pre_part_2=self.get_my_train_data()
        # print("model_pre_part_2 starting.......")
        # with torch.cuda.device("cuda:0"):
        #     for i in range(100):
        #         self.my_train_part_2(train_loader_pre_part_2,model_pre_part_2,criterion_1,optimizer_pre_part_2,i,model_src,model_tgt,model_pre_part)
        # model_pre_part_2.eval()
        
        # if os.path.exists(self.model_pre_part_2_path):
        #     model_pre_part_2=torch.load(self.model_pre_part_2_path)
        # else:
        #     train_loader_pre_part_2=self.get_my_train_data()

        #     model_pre_part_2=Attention(vector_dim=self.emb_dim,num_heads=4,num_layers=1,feedforward_dim=1024)
        #     model_pre_part_2=model_pre_part_2.cuda() 

        #     optimizer_pre_part_2=torch.optim.Adam(params=model_pre_part_2.parameters(), lr=self.lr, weight_decay=self.wd)
                
        #     print("model_pre_part_2 starting.......")
        #     with torch.cuda.device("cuda:0"):
        #         for i in range(10):
        #             self.my_train_part_2(train_loader_pre_part_2,model_pre_part_2,criterion_1,optimizer_pre_part_2,i,model_src,model_tgt,model_pre_part)
        #     # torch.save(model_pre_part_2,self.model_pre_part_2_path)
        # model_pre_part_2.eval()
#-----------------------------pre train_part_2------------------------------------------   
        print("-----------------------------------------------------------------------")
        print("-----------------------------------------------------------------------")
        print("-----------------------------------------------------------------------")
        batch_size=256
        temperature=0.02
        print("temperature:",temperature,"batch_size:",batch_size)
        print("-----------------------------------------------------------------------")
        print("-----------------------------------------------------------------------")
        print("-----------------------------------------------------------------------")
        train_loader_pre=self.get_my_data()
            
        print('=====CDR Pretraining=====')
        with torch.cuda.device("cuda:0"):
            #print("mlp(encoder) layer num:"+str(self.layers_num))
            # print("lr_pre:"+str(self.lr_pre)+"  "+"wd_pre:"+str(self.wd_pre))
            simclr = SimCLR(model=model_pre, optimizer=optimizer_pre,model_src=model_src,model_tgt=model_tgt,model_pre_part=model_pre_part,temperature=0.02,batch_size=256)
            simclr.train(train_loader_pre)
    #-----------------------------pre train------------------------------------------

    #-------------------------------train----------------------------------------------
        model_pre.eval()

        model = Emcdr(self.emb_dim)
        model = model.cuda()


        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr, weight_decay=self.wd)

        train_loader=self.get_my_train_data()  #需要进行更改
        test_loader=self.get_my_test_data()

        with torch.cuda.device("cuda:0"):
            for i in range(self.epoch):
                self.my_train(train_loader,model,criterion_1,optimizer,i,model_src,model_tgt,model_pre)
                mae, rmse = self.eval_mae(model, test_loader,model_src,model_tgt,model_pre)
                #self.my_train(train_loader,model,criterion_1,optimizer,i,model_src,model_tgt)
                #mae, rmse = self.eval_mae(model, test_loader,model_src,model_tgt)
                self.update_results(mae, rmse, 'emcdr')
                print('MAE: {} RMSE: {}'.format(mae, rmse))
        print("temperature:",temperature,"batch_size:",batch_size)
        print("final result:"+'MAE: {} RMSE: {}'.format(self.results['emcdr_mae'], self.results['emcdr_rmse']))
#-------------------------------train----------------------------------------------
