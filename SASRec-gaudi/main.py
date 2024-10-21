import os
import time
import torch
import argparse
import numpy as np

###GAUDI
import habana_frameworks.torch.core as htcore

from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

## Debug
import sys
from habana_utils import set_seed, SEED

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
# parser.add_argument('--dataset', default = 'Amazon_Fashion')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='hpu', type=str, help='cpu, hpu')
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--nn_parameter', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--sampling', default=0, type=int, help='sampling rate, 0 = non sample')

global_step=0

def pre_hook(module, args, kwargs):
    print(f'PRE HOOK! for module {module}, step {global_step}')
    print(args)
    print(kwargs)
    print(f'PRE HOOK! for module {module}, step {global_step}')


def hook(module, args, output):
    print(f'POST HOOK! for module {module}, step {global_step}')
    print("******args*****")
    print(args)
    print("******output*****")
    print(output)
    print(f'POST HOOK! for module {module}, step {global_step}')


def pre_bk_hook(module, grad_output):
    print(f'PRE BK HOOK! for module {module}, step {global_step}')
    print(f"******grad_output*********")
    print(grad_output)
    print(f'PRE BK HOOK! for module {module}, step {global_step}')


def bk_hook(module, grad_input, grad_output):
    print(f'POST BK HOOK! for module {module}, step {global_step}')
    print(f"-------grad_input-----")
    print(grad_input)
    print(f"-------grad_output-----")
    print(grad_output)
    print(f'POST BK HOOK! for module {module}, step {global_step}')

args = parser.parse_args()

if __name__ == '__main__':
    
    set_seed(SEED)
    # global dataset
    
    if (not os.path.isfile(f'./data/processed/{args.dataset}_train.txt')) or (not os.path.isfile(f'./data/processed/{args.dataset}_valid.txt') or (not os.path.isfile(f'./data/processed/{args.dataset}_test.txt'))):
        print("Download Dataset")
        preprocess(args.dataset)
    dataset = data_partition(args.dataset, args)
    
    
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    if args.device =='hpu':
        args.device = torch.device('hpu')
    if args.device =='cpu':
        args.device = torch.device('cpu')
    
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device)
    print(model)
    
    # Moving xavier_normal to cpu
    if args.device.type == "hpu":
        model = model.to(torch.device("cpu"))

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    if args.device.type == "hpu":
        model = model.to(args.device)
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device))
            kwargs['args'].device = args.device
            model = SASRec(**kwargs).to(args.device)
            model.load_state_dict(checkpoint)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    time_list = []
    loss_list = []
    T = 0.0
    t0 = time.time()
    start_time = time.time()
    
    # Apply debug hooks
    for name, module in model.named_modules():
        print(f"Adding hook for {name} module {module}")
        module.register_forward_hook(hook)
        module.register_forward_pre_hook(pre_hook,with_kwargs=True)
        module.register_full_backward_hook(bk_hook)
        module.register_full_backward_pre_hook(pre_bk_hook)


    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        model.train()
        epoch_s_time = time.time()
        total_loss, count = 0, 0
        if args.inference_only: break
        # Fix steps to 5 for debug
        for step in range(5):
            print(f"step is {step}")
            global_step = step
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            
            #nn.Embedding
            if args.nn_parameter:
                loss += args.l2_emb * torch.norm(model.item_emb)
            else:
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
             
            #GAUDI
            loss.backward()
            if args.device.type =='hpu':
                htcore.mark_step()
            adam_optimizer.step()
            if args.device.type =='hpu':
                htcore.mark_step()
            
            total_loss += loss.item()
            count+=1
            
            if step % 100 == 0:
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        
        epoch_e_time = time.time()
        time_list.append(epoch_e_time - epoch_s_time)
        loss_list.append(total_loss/count)
    
        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            with torch.no_grad():
                t_test = evaluate(model, dataset, args)
                t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train()
    
        if epoch == args.num_epochs:
            folder = args.dataset
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)):
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname))
    
    sampler.close()
    end_time = time.time()
    print("Done")
    print("Time:", end_time-start_time)
    
    np.save('time.npy', np.array(time_list))
    np.save('loss.npy', np.array(loss_list))