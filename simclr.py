import torch
import torch.nn.functional as F
from tqdm import tqdm

torch.manual_seed(0)

# self.device=torch.self.device("cpu")

class SimCLR(object):

    def __init__(self, **kwargs):
        self.device=torch.device("cuda:0")
        self.pre_model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.model_src=kwargs['model_src'].to(self.device)
        self.model_tgt=kwargs['model_tgt'].to(self.device)
        self.model_pre_part=kwargs['model_pre_part'].to(self.device)
        # self.model_pre_part_2=kwargs['model_pre_part_2'].to(self.device)
        self.criterion_1 = torch.nn.CrossEntropyLoss().to(self.device)
        self.criterion_2 = torch.nn.MSELoss().to(self.device)
        self.epochs=150
        self.batch_size=kwargs['batch_size']
        self.temperature=kwargs['temperature']

    def info_nce_loss(self, features):       
        current_batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(current_batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)       #新的这个矩阵的形状是 similarity_matrix.shape[0]*（similarity_matrix.shape[1]-1）
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse


    def train(self, train_loader):
        print("SimCLR batchsize:"+str(self.batch_size))
        print("SimCLR epochs:"+str(self.epochs))
        print("temperture:"+str(self.temperature))
        for epoch in range(self.epochs):
            self.pre_model.train()
            print('Training Epoch {}:'.format(epoch + 1))
            for user_id,_ in tqdm(train_loader):
                user_embeddings_src=self.model_src.uid_embedding(user_id.unsqueeze(1)).squeeze().to(self.device)
                # user_embeddings_tgt=self.model_tgt.uid_embedding(user_id.unsqueeze(1)).squeeze().to(self.device)
                user_embeddings_tgt_pred=self.model_pre_part(user_embeddings_src)
                # user_embeddings_tgt_final=self.model_pre_part_2(user_embeddings_tgt_pred,user_embeddings_tgt)

                # src_key_padding_mask=torch.cat(src_key_padding_mask_1,src_key_padding_mask_2,dim=0)
                # user_embeddings_src_tgt = torch.cat((user_embeddings_src, user_embeddings_tgt_final), dim=0).to(self.device)
                # user_embeddings_src_tgt = user_embeddings_src_tgt.to(self.device)

                features_user_embeddings_src = self.pre_model(user_embeddings_src).to(self.device)
                # features_user_embeddings_tgt = self.pre_model(user_embeddings_tgt).to(self.device)

                features=torch.cat((features_user_embeddings_src,user_embeddings_tgt_pred),dim=0).to(self.device)

                logits, labels = self.info_nce_loss(features)
                # loss = 0.5*self.criterion_1(logits, labels)+average_loss
                loss=self.criterion_1(logits,labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            


            #测试指标

                
#把这段代码直接转移到了train的代码这当中了

    # def eval_mae(self,data_loader):
    #     print('Evaluating MAE:')
    #     model.eval()
    #     targets, predicts = list(), list()
    #     loss = torch.nn.L1Loss()
    #     mse_loss = torch.nn.MSELoss()
    #     with torch.no_grad():
    #         for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
    #             pred = model(X, stage)
    #             targets.extend(y.squeeze(1).tolist())
    #             predicts.extend(pred.tolist())
    #     targets = torch.tensor(targets).float()
    #     predicts = torch.tensor(predicts)
    #     return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()
