import json
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
from .utils import calculate_auc, setup_features, sample_edges
from sklearn.model_selection import train_test_split
from .original_signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep, ListModule

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network. Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """
    def __init__(self, device, args, X, nodes_dict):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.nodes_dict = nodes_dict
        self.setup_layers()
        
        
        #  node under sampling
#         if self.args.node_under_sampling:
#             less_class_num = np.unique(node_labels,return_counts=True)[1].min()
#             sampled_indice = np.append(np.random.choice(np.where(node_labels==1)[0],less_class_num,replace=False),np.random.choice(np.where(node_labels==-1)[0],less_class_num,replace=False))
#             self.train_target_indice, self.test_target_indice = train_test_split(sampled_indice,test_size=args.test_size,random_state=0)
#         else:
#             self.train_target_indice, self.test_target_indice = train_test_split(np.arange(len(node_indice)),test_size=args.test_size,random_state=0)
#         self.train_z_indice = node_indice[self.train_target_indice]
#         self.test_z_indice = node_indice[self.test_target_indice]


    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers and Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.positive_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
        self.negative_base_aggregator = SignedSAGEConvolutionBase(self.X.shape[1]*2, self.neurons[0]).to(self.device)
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1,self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3*self.neurons[i-1], self.neurons[i]).to(self.device))
        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        # self.regression_weights = Parameter(torch.Tensor(4*self.neurons[-1], 3))
        self.regression_weights = Parameter(torch.Tensor(2*self.neurons[-1], 1))
        init.xavier_normal_(self.regression_weights)

    def calculate_regression_loss(self, z, target, train_indice):
        train_z_indice = self.nodes_dict['indice'][train_indice]
        train_z = z[train_z_indice]
        predictions = torch.mm(train_z,self.regression_weights)
        # predictions_soft = F.log_softmax(predictions,dim=1)
        predictions_sigmoid = torch.sigmoid(predictions)
        train_target = target[train_indice]
        if self.args.class_weights:
            class_counts = np.unique(train_target.cpu().numpy(),return_counts=True)[1]
            weight = torch.Tensor([class_counts[1],class_counts[0]]).to(self.device)
        else:
            weight = None
        # loss_term = F.nll_loss(predictions_soft,train_target,weight=weight)
        train_target = train_target.float()
        loss_term = F.binary_cross_entropy(predictions_sigmoid.squeeze(),train_target,weight=weight)
        return loss_term, predictions_sigmoid



    def calculate_positive_embedding_loss(self, z, positive_edges):
        """
        Calculating the loss on the positive edge embedding distances
        :param z: Hidden vertex representation.
        :param positive_edges: Positive training edges.
        :return loss_term: Loss value on positive edge embedding.
        """
        self.positive_surrogates = [random.choice(self.nodes) for node in range(positive_edges.shape[1])]
        self.positive_surrogates = torch.from_numpy(np.array(self.positive_surrogates, dtype=np.int64).T).type(torch.long).to(self.device)
        positive_edges = torch.t(positive_edges)
        self.positive_z_i, self.positive_z_j = z[positive_edges[:,0],:],z[positive_edges[:,1],:]
        self.positive_z_k = z[self.positive_surrogates,:]
        norm_i_j = torch.norm(self.positive_z_i-self.positive_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.positive_z_i-self.positive_z_k, 2, 1, True).pow(2)
        term = norm_i_j-norm_i_k
        term[term<0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_negative_embedding_loss(self, z, negative_edges):
        """
        Calculating the loss on the negative edge embedding distances
        :param z: Hidden vertex representation.
        :param negative_edges: Negative training edges.
        :return loss_term: Loss value on negative edge embedding.
        """
        self.negative_surrogates = [random.choice(self.nodes) for node in range(negative_edges.shape[1])]
        self.negative_surrogates = torch.from_numpy(np.array(self.negative_surrogates, dtype=np.int64).T).type(torch.long).to(self.device)
        negative_edges = torch.t(negative_edges)
        self.negative_z_i, self.negative_z_j = z[negative_edges[:,0],:], z[negative_edges[:,1],:]
        self.negative_z_k = z[self.negative_surrogates,:]
        norm_i_j = torch.norm(self.negative_z_i-self.negative_z_j, 2, 1, True).pow(2)
        norm_i_k = torch.norm(self.negative_z_i-self.negative_z_k, 2, 1, True).pow(2)
        term = norm_i_k-norm_i_j
        term[term<0] = 0
        loss_term = term.mean()
        return loss_term

    def calculate_loss_function(self, z, positive_edges, negative_edges, target, train_indice):
        """
        Calculating the embedding losses, regression loss and weight regularization loss.
        :param z: Node embedding.
        :param positive_edges: Positive edge pairs.
        :param negative_edges: Negative edge pairs.
        :param target: Target vector.
        :return loss: Value of loss.
        """
        loss_term_1 = 0 #self.calculate_positive_embedding_loss(z, positive_edges)
        loss_term_2 = 0 #self.calculate_negative_embedding_loss(z, negative_edges)
        regression_loss, self.predictions = self.calculate_regression_loss(z,target,train_indice)
        l1_loss = 0
        for param in self.regression_weights:
            l1_loss += torch.norm(param,1)
        l1_loss = l1_loss / self.regression_weights.shape[0]
        
        loss_term = regression_loss+self.args.lamb*(loss_term_1+loss_term_2) + self.args.l1_lambda*l1_loss
        return loss_term

    def forward(self, positive_edges, negative_edges, target, train_indice):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        :param positive_edges: Positive edges.
        :param negative_edges: Negative edges.
        :param target: Target vectors.
        :return loss: Loss value.
        :return self.z: Hidden vertex representations.
        """
        self.h_pos, self.h_neg = [],[]
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1,self.layers):
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1],self.h_neg[i-1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(self.negative_aggregators[i-1](self.h_neg[i-1],self.h_pos[i-1], positive_edges, negative_edges)))
        if self.args.hidden_residual is False:
            self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1)
        else:
            self.z = torch.cat((self.h_pos[-1],self.h_neg[-1]), 1) + torch.cat((self.h_pos[0],self.h_neg[0]),1)
        loss = self.calculate_loss_function(self.z, positive_edges, negative_edges, target, train_indice)
        return loss, self.z

class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """
    def __init__(self, args, edges,nodes_dict):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges
        self.nodes_dict = nodes_dict
        self.node_labels = nodes_dict['label']
        self.node_indice = nodes_dict['indice']
        self.node_count = nodes_dict['all_ncount']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logs()


    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] =  vars(self.args)
        self.logs["performance"] = [["Epoch","AUC","F1"]]
        self.logs["training_time"] = [["Epoch","Seconds"]]


    def setup_dataset(self):
        """
        Creating train and test split.
        """
        # self.positive_edges, self.test_positive_edges = train_test_split(self.edges["positive_edges"], test_size = self.args.test_size)
        # self.negative_edges, self.test_negative_edges = train_test_split(self.edges["negative_edges"], test_size = self.args.test_size)
        self.positive_edges = self.edges["positive_edges"]
        self.negative_edges = self.edges["negative_edges"]

        self.ecount = len(self.positive_edges + self.negative_edges)
        node_count =  self.node_count # 
        self.X = setup_features(self.args, self.positive_edges, self.negative_edges, node_count)
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        # self.y = np.array([0 if i< int(self.ecount/2) else 1 for i in range(self.ecount)] +[2]*(self.ecount*2))
        # self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        # self.y :
        self.y = np.array([1 if label==-1 else 0 for label in self.node_labels])
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)

        self.X = torch.from_numpy(self.X).float().to(self.device)


    def score_model(self, epoch):
        loss, self.train_z = self.model(self.positive_edges, self.negative_edges, self.y,self.train_indice)
        
        test_z_indice = self.nodes_dict['indice'][self.test_indice]
        test_hidden = self.train_z[test_z_indice]
        scores = torch.mm(test_hidden,self.model.regression_weights.to(self.device))
        # probability_scores = torch.exp(F.softmax(scores, dim=1))
        # predictions = F.softmax(scores,dim=1)
        predictions = torch.sigmoid(scores)
        predictions = predictions.cpu().detach().numpy()
        test_target = self.y[self.test_indice]
        test_target = test_target.cpu().detach().numpy()
        auc, f1 = calculate_auc(test_target,predictions,self.edges)
        self.logs["performance"].append([epoch+1, auc, f1])
        return auc

    def create_and_train_model(self):
        """
        Model training and scoring.
        """
        print("\nTraining started.\n")
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X, self.nodes_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        self.epochs = trange(self.args.epochs, desc="Loss")
        best_auc_score = 0.0
        all_train_indice, test_indice = train_test_split(np.arange(len(self.node_indice)),test_size=self.args.test_size,stratify=self.node_labels)
        for epoch in self.epochs:
            start_time = time.time()
            self.optimizer.zero_grad()
            self.train_indice, self.test_indice = self.under_sampling(all_train_indice), test_indice
#             if self.args.sample_num is not None:
#                 self.sampled_positive_edges, self.sampled_negative_edges = sample_edges(self.edges,self.args.sample_num,self.nodes_dict)
#                 self.sampled_positive_edges = torch.from_numpy(np.array(self.sampled_positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
#                 self.sampled_negative_edges = torch.from_numpy(np.array(self.sampled_negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
#                 loss, _ = self.model(self.sampled_positive_edges, self.sampled_negative_edges, self.y)
#             else:
#                 loss, _ = self.model(self.positive_edges, self.negative_edges, self.y)
            loss, _ = self.model(self.positive_edges, self.negative_edges, self.y, self.train_indice)
            loss.backward()
            self.epochs.set_description("SGCN (Loss=%g)" % round(loss.item(),4))
            self.optimizer.step()
            self.logs["training_time"].append([epoch+1,time.time()-start_time])
            if (self.args.test_size >0) and (epoch % self.args.eval_freq == 0): 
                current_auc = self.score_model(epoch)
                if current_auc >= best_auc_score:
                    best_auc_score = current_auc
                    self.save_model()
                    if self.args.inductive_model_path is not None:
                        torch.save(self.model.state_dict(), self.args.inductive_model_path)
                

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        # print("\nEmbedding is saved.\n")
        self.train_z = self.train_z.cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate([np.array(range(self.train_z.shape[0])).reshape(-1,1),self.train_z],axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns = embedding_header)
        # self.train_z.to_csv(self.args.embedding_path, index = None)
        self.train_z.to_pickle(self.args.embedding_path)
        # print("\nRegression weights are saved.\n")
        self.regression_weights = self.model.regression_weights.cpu().detach().numpy().T
        regression_header = ["x_" + str(x) for x in range(self.regression_weights.shape[1])]
        self.regression_weights = pd.DataFrame(self.regression_weights, columns = regression_header)
        # self.regression_weights.to_csv(self.args.regression_weights_path, index = None)
        self.regression_weights.to_pickle(self.args.regression_weights_path)
        
    def under_sampling(self,all_train_indice):
        all_train_target = self.node_labels[all_train_indice]
        if self.args.node_under_sampling:
            less_cls_num = np.unique(all_train_target,return_counts=True)[1].min()
            sampled_indice = np.append(np.random.choice(np.where(all_train_target==1)[0],less_cls_num,replace=False),
                                       np.random.choice(np.where(all_train_target==-1)[0],less_cls_num,replace=False))
            return sampled_indice
        else:
            return all_train_indice
        
        
class SignedGCNPredictor(object):
    """
    Object to classify the users with the input of X and edges in the inductive manner.
    """
    def __init__(self, args, inductive_model_path, X, edges,nodes_dict):
        self.args = args
        self.inductive_model_path = inductive_model_path
        self.X = X

        self.edges = edges
        self.positive_edges = self.edges["positive_edges"]
        self.negative_edges = self.edges["negative_edges"]

        
        self.node_indice = nodes_dict['indice']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges, dtype=np.int64).T).type(torch.long).to(self.device)
        self.X = torch.from_numpy(self.X).float().to(self.device)
        
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X, nodes_dict).to(self.device)
        self.model.load_state_dict(torch.load(inductive_model_path))


    def predict(self):
        self.y = np.zeros(self.X.shape[0])
        self.y = torch.from_numpy(self.y).type(torch.LongTensor).to(self.device)
        _, self.z = self.model(self.positive_edges, self.negative_edges, self.y,np.array([1,2,3])) #
        scores = torch.mm(self.z,self.model.regression_weights.to(self.device))
        predictions = torch.sigmoid(scores)
        predictions = predictions.cpu().detach().numpy()
        return predictions
           
