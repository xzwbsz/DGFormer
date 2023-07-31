import torch

class Seq2SeqAttrs:
    def __init__(self, sparse_idx, angle_ratio, geodesic, config):
        self.sparse_idx = sparse_idx
        self.max_view = int(config['clcrn']['max_view'])
        self.cl_decay_steps = int(config['clcrn']['cl_decay_steps'])
        self.node_num = int(config['clcrn']['node_num'])
        self.layer_num = int(config['clcrn']['layer_num'])
        self.rnn_units = int(config['clcrn']['rnn_units'])
        self.input_dim = int(config['clcrn']['input_dim'])
        self.output_dim = int(config['clcrn']['output_dim'])
        self.seq_len = int(config['train']['hist_len'])
        self.lck_structure = [4,8]
        self.embed_dim = int(config['clcrn']['embed_dim'])
        self.location_dim = int(config['clcrn']['location_dim'])
        self.horizon = int(config['train']['pred_len'])
        self.hidden_units = int(config['clcrn']['hidden_units'])
        self.block_num = int(config['clcrn']['block_num'])
        angle_ratio = torch.sparse.FloatTensor(
            self.sparse_idx, 
            angle_ratio, 
            (self.node_num,self.node_num)
            ).to_dense() 
        self.angle_ratio = angle_ratio + torch.eye(*angle_ratio.shape).to(angle_ratio.device)
        self.geodesic =  torch.sparse.FloatTensor(
            self.sparse_idx, 
            geodesic, 
            (self.node_num,self.node_num)
            ).to_dense()