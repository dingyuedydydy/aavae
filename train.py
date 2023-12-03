from evaluator import Evaluator
import json
import torch
import torch.utils.data as Data
import model_transformer2
import model_VSAN
import dataset_load
import numpy as np
import os
import logging
import traceback
import loss_func
import random
import time
import argparse
import sys
from tensorboardX import SummaryWriter

# Hyper Parameters
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# use GPU or CPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

hyper_params = {
    # dataset to use (datasets/) [ml-1m, ml-latest, ml-10m] provided
    'dataset_path': 'music',
    'seq_len': 50,
    # alpha
    'kl_weight': 0.5,
    # beta
    # 'contrast_weight': 0.1,
    'infonce_weight': 0.5,
    # Total epochs
    'epochs': 350,
    # Set the number of users in evaluation during training to speed up
    # If set to None, all of the users will be evaluated
    'evaluate_users': None,
    'item_embed_size': 128,
    'rnn_size': 100,
    'hidden_size': 100,
    'latent_size': 64,
    'timesteps': 5,
    'test_prop': 0.2,
    'batch_size': 64,
    'anneal': False,
    'time_split': True,
    'model_func': 'attention',
    'add_eps': True,
    'device': device,
    'check_freq': 4,
    'Ks': [5, 10, 20],
    'lr_primal': 1e-4,
    'lr_dual': 1e-05,
    'lr_prior': 5e-4,
    'l2_regular': 1e-2,
    'l2_adver': 1e-1,
    'total_step': 2000000,
    'N_of_layers': 3,
    'N_of_attentions': 4,
    'd_model': 128,
    'd_ff': 2048,
    'heads': 1,
    'dropout': 0.4
}

hyper_params['total_users'], hyper_params['total_items'], _ = dataset_load.count_data(hyper_params['dataset_path'])

info_str = 'dataset:' + hyper_params['dataset_path'] + ' lr_primal:' + str(hyper_params["lr_primal"]) + ' lr_prior:' + str(
    hyper_params['lr_dual']) + ' lr_prior:' + str(hyper_params["lr_prior"])  + ' l2_regular:' + str(hyper_params["l2_regular"])  + ' l2_adver:' + str(hyper_params["l2_adver"])  + ' kl:' + str(hyper_params['kl_weight']) + ' infonce:' + str(
    hyper_params['infonce_weight']) + ' batch:' + str(hyper_params['batch_size']) + ' attention_blocks:' + str(hyper_params[
               'N_of_layers']) + ' atentions:' + str(hyper_params['N_of_attentions'])
path_str = f'{hyper_params["model_func"]}_{hyper_params["dataset_path"]}_kl_{hyper_params["kl_weight"]}_contrast_{hyper_params["infonce_weight"]}_dropout_{hyper_params["dropout"]}_dmodel_{hyper_params["d_model"]}'

# Parser
parser = argparse.ArgumentParser(prog='train')
parser.add_argument("-m", "--msg", default=hyper_params['dataset_path'])
args = parser.parse_args()
train_msg = args.msg


# Setup Seed.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1111)

# Config logging module.
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
local_time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
handler = logging.FileHandler(
    "model_log/log_" + local_time_str + '_' + train_msg.replace(' ', '_') + ".txt")

handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(train_msg)
logger.info(info_str)
# commented when use cpu for training
# logger.info('Using CUDA:' + os.environ['CUDA_VISIBLE_DEVICES'])

# Accelerate with CuDNN.
# torch.backends.cudnn.benchmark = True

# Load data at first.
dataset_load.load_data(hyper_params)
user_dataset = dataset_load.generate_train_data(hyper_params)
user_dataloader = Data.DataLoader(
    user_dataset, batch_size=hyper_params['batch_size'], shuffle=True)
test_dataset = dataset_load.generate_test_data(hyper_params)
test_dataloader = Data.DataLoader(
    test_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

# Generate validate dataset.
val_dataset = dataset_load.generate_validate_data(hyper_params)
val_dataloader = Data.DataLoader(
    val_dataset, batch_size=hyper_params['batch_size'], shuffle=True)

# Build the model.
print('Building net...')
logger.info('Building net...')

# # use gpu to train
# net = model.Model(hyper_params).to(hyper_params['device'])
# adversary = model.Adversary(hyper_params).to(hyper_params['device'])
# contrast_adversary = model.GRUAdversary(
#     hyper_params).to(hyper_params['device'])

# use cpu to train
net = model_transformer2.Model(hyper_params).to(device)
adversary = model_transformer2.Adversary(hyper_params).to(device)
contrast_adversary = model_transformer2.GRUAdversary(
    hyper_params).to(device)

print(net)
print('Net build finished.')
logger.info('Net build finished.')

# ---------------------------------------Optimizer-----------------------------------------
optimizer_primal = torch.optim.AdamW([{
    'params': net.parameters(),
    'lr': hyper_params['lr_primal'],
    'weight_decay': hyper_params['l2_regular']}
])
optimizer_dual = torch.optim.SGD([{
    'params': net.model.encoder.parameters(),
    'lr': hyper_params['lr_dual'],
    'weight_decay': hyper_params['l2_adver']}
])
optimizer_prior = torch.optim.SGD([{
    'params': adversary.parameters(),
    'lr': hyper_params['lr_prior'],
    'weight_decay': hyper_params['l2_adver']}
])

print('User datasets loaded and saved.')
logger.info('User datasets loaded and saved.')

# Evaluator
evaluator = Evaluator(hyper_params=hyper_params, logger=logger)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    #print(subsequent_mask.shape)
    return torch.from_numpy(subsequent_mask) == 0


# def make_std_mask(tgt):
#     """Create a mask to hide padding and future words."""
#     tgt_mask = (tgt != 0).unsqueeze(-2)
#     #print(tgt_mask.dtype, tgt_mask.shape)
    
#     kkk = subsequent_mask(tgt.size(-1))
#     #print("kkk",kkk.dtype,kkk.shape)
#     mmm = torch.autograd.Variable(kkk.type_as(tgt))
#     #print("mmm",mmm.dtype,mmm.shape)
#     tgt_mask = tgt_mask & mmm
#     #print("tgt_mask",tgt_mask.shape)
#     return tgt_mask

def make_std_mask(tgt):
    """Create a mask to hide padding and future words."""
    tgt_mask = (tgt != 0).unsqueeze(-2)
    tgt_mask = tgt_mask & torch.autograd.Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt))
    return tgt_mask

def corrcoef(x):
        """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
        f = (x.shape[0] - 1) / x.shape[0]      # 方差调整系数
        x_reducemean = x - torch.mean(x, axis=0)
        #print(x_reducemean)
        numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
        #print(numerator)
        var_ = x.var(axis=0).reshape(x.shape[1], 1)
        denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
        #print(denominator)
        corrcoef = torch.div(numerator, denominator)
        eye = torch.eye(numerator.shape[0])
        corrcoef2 = corrcoef * corrcoef
        corrcoef2 = corrcoef2 -  torch.eye(corrcoef2.shape[0]).to(device)

        return corrcoef2.mean()


def cov(input):
    
    b, c, h, w = input.size()
    x = input- torch.mean(input)
    x = x.view(b * c, h * w)
    cov_matrix = torch.matmul(x.T, x) / x.shape[0]
    cov_matrix = cov_matrix -  torch.eye(cov_matrix.shape[0])
    return cov_matrix.mean()

def train():
    writer = SummaryWriter(f'./runs/{path_str}')
    print('Start training...')
    logger.info('Start training...')

    global_step = 0
    mebank = loss_func.MetricShower()

    for epoch in range(hyper_params['epochs']):
        net.train()
        corr_sum = 0
        varience_sum = 0
        count = global_step
        for batchx, batchy, padding, user_id, cur_cnt in user_dataloader:
            # batchx = batchx.to(hyper_params['device'])
            # batchy = batchy.to(hyper_params['device'])
            # padding = padding.to(hyper_params['device'])
            # user_id = user_id.to(hyper_params['device'])
            # cur_cnt = cur_cnt.to(hyper_params['device'])

            batchx = batchx.to(device)
            mask = make_std_mask(batchx)
            batchy = batchy.to(device)
            padding = padding.to(device)
            user_id = user_id.to(device)
            cur_cnt = cur_cnt.to(device)

            # Forward.
            optimizer_primal.zero_grad()
            # optimizer_dual.zero_grad()
            pred, x_real, z_inferred, out_embed = net(batchx, mask, hyper_params['add_eps'])
            # print("pred:",pred.shape)
            # print("batchy:",batchy.shape)

            # --------------------------VAE---------------------------
            multi_loss = loss_func.vae_loss(
                pred, batchy, padding, hyper_params)
            if hyper_params['anneal']:
                anneal = global_step / \
                         hyper_params['total_step'] * hyper_params['kl_weight']
            else:
                anneal = hyper_params['kl_weight']

            kl_loss = loss_func.kl_loss(
                adversary, x_real, z_inferred, padding, KL_WEIGHT=anneal)

            query = out_embed[:, hyper_params['seq_len'] - 1, :]
            query = query.squeeze(-2)

            positive_key = z_inferred[:, hyper_params['seq_len']//2 - 1, :]
            positive_key = positive_key.squeeze(-2)
            negative_keys = query

            info_nce_loss = loss_func.info_nce_loss(query=query, positive_key=positive_key, negative_keys=negative_keys, INFONCE_WEIGHT=hyper_params['infonce_weight'])

            loss = multi_loss + info_nce_loss + kl_loss
            loss.backward()
            optimizer_primal.step()
            # optimizer_dual.step()

            # --------------------------ADVER------------------------------
            optimizer_prior.zero_grad()
            adver_kl_loss = loss_func.adversary_kl_loss(
                adversary, x_real.detach(), z_inferred.detach(), padding)
            adver_kl_loss.backward()
            optimizer_prior.step()

            mebank.store({'vae': multi_loss.item(), 'kl': kl_loss.item(), 'prior': adver_kl_loss.item(), 'infoNCE': info_nce_loss.item()})
            corr_sum += corrcoef(query).item()
            global_step += 1
            query2 = out_embed[:, hyper_params['seq_len'] - 1, :]
            query2 = query2.squeeze(-2)
            varience_sum += query2.std() * query2.std()

        mebank.store({'corr': corr_sum / (global_step - count), 'vari': varience_sum / (global_step - count)})    
        # Show Loss
        print(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},infoNCE:{mebank.get("infoNCE")},prior:{mebank.get("prior")},corr:{mebank.get("corr")},vari:{mebank.get("vari")}')
        logger.info(
            f'EPOCH:({epoch}/{hyper_params["epochs"]}),STEP:{global_step}/{hyper_params["total_step"]},vae:{mebank.get("vae")},kl:{mebank.get("kl")},infoNCE:{mebank.get("infoNCE")},prior:{mebank.get("prior")},corr:{mebank.get("corr")},vari:{mebank.get("vari")}')

        writer.add_scalar('loss', mebank.get("vae"), global_step=epoch)
        writer.flush()

        mebank.clear()

        # Check (These codes are just to monitor the results of training)
        # Here val_dataloader and test_dataloader are the same
        if epoch % hyper_params['check_freq'] == 0:
            hr, _, _ = evaluator.evaluate(net, adversary, dataloader=val_dataloader,
                                          validate=True, evaluate_users=hyper_params['evaluate_users'])
            writer.add_scalar('hr10', hr[2], global_step=epoch)
            writer.flush()
            net.train()
            adversary.train()

        if global_step >= hyper_params['total_step']:
            break

    evaluator.evaluate(net, adversary, dataloader=test_dataloader,
                       validate=False)
    writer.close()


# Main
if __name__ == '__main__':
    # Train the model.
    try:
        train()
        logger.info('Finished.')
    except Exception as err:
        err_info = traceback.format_exc()
        print(err_info)
        logger.info(err_info)
        logger.info('Error.')
