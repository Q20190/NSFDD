from __future__ import division
import argparse
import random
import torch
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from gycutils.utils import make_trainable,calc_gradient_penalty,calculate_metric_percase
from gan import discriminator,generator
from torch.optim import Adam
from loss import BCE_Loss
from torchvision import utils
import tqdm
from PIL import Image
import numpy as np
import os
from gycutils.dataset_synapse import Synapse_dataset, RandomGenerator
from pydensecrf.densecrf import DenseCRF,DenseCRF2D
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ceshi', help='experiment_name')
parser.add_argument('--n_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=110, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.00298,
                    help='segmentation network learning rate')
parser.add_argument('--input_chanel', type=int,default=1)
parser.add_argument('--img_size', type=int,
                    default=256, help='input size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=2, help='using number of skip-connect, default is num')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--decoder_channels', type=list,default=[512, 256,128, 64])
parser.add_argument('--skip_channels', type=list,default=[512, 256,128, 64])
parser.add_argument('--activation', type=str,default='softmax')
parser.add_argument('--attn_type', type=str,default='DA')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

trainloader = DataLoader(Synapse_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size],split='train')])),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

valloader = DataLoader(Synapse_dataset(base_dir=args.root_path, split="test",transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size],split='test')])),
                                            batch_size=1, shuffle=False, num_workers=0)

D=torch.nn.DataParallel(discriminator(n_filters=32)).cuda()
G=torch.nn.DataParallel(generator(args,args.n_classes ,n_filters=32)).cuda()
gan_loss_percent=0.03

one=torch.FloatTensor([1])
mone=one*-1
moneg=one*-1*gan_loss_percent

one=one.cuda()
mone=mone.cuda()
moneg=moneg.cuda()

loss_func=BCE_Loss()
optimizer_D=Adam(D.parameters(),lr=args.base_lr,betas=(0.5,0.9),eps=10e-8)
optimizer_G=Adam(G.parameters(),lr=args.base_lr,betas=(0.5,0.9),eps=10e-8)

for epoch in range(args.max_epochs):
    D.train()
    G.train()
    #train D
    make_trainable(D,True)
    make_trainable(G,False)
    for idx,sample_batch in enumerate(trainloader):
        real_imgs=sample_batch['image'].unsqueeze(1).cuda()
        real_labels=sample_batch['label'].unsqueeze(1).cuda()
        D.zero_grad()
        optimizer_D.zero_grad()

        real_pair = torch.cat((real_imgs, real_labels), dim=1)
        #real_pair_y=torch.ones((real_pair.size()[0],1)).cuda()
        d_real = D(real_pair)
        d_real = d_real.mean()
        d_real.backward(mone[0])

        fake_images=torch.sigmoid(G(real_imgs))
        fake_mask = (fake_images > 0.5).long()

        fake_pair=torch.cat((real_imgs, fake_images), dim=1)
        #fake_pair_y=torch.zeros((real_pair.size()[0],1)).cuda()
        d_fake=D(fake_pair)
        d_fake=d_fake.mean()
        d_fake.backward(one[0])

        #d_loss=loss_func(D(real_pair),real_pair_y)+loss_func(D(fake_pair),fake_pair_y)
        #d_loss.backward()
        gradient_penalty=calc_gradient_penalty(D,real_pair.data,fake_pair.data)
        gradient_penalty.backward(retain_graph=True)

        Wasserstein_D=d_real-d_fake
        optimizer_D.step()
    #train G
    make_trainable(D,False)
    make_trainable(G,True)
    for idx,sample_batch in enumerate(trainloader):
        G.zero_grad()
        optimizer_G.zero_grad()
        real_imgs=sample_batch['image'].unsqueeze(1).cuda()
        real_labels=sample_batch['label'].unsqueeze(1).cuda()
        logits=G(real_imgs)
        pred_labels=torch.sigmoid(logits)
        pred_mask = (pred_labels > 0.5).long()
        Seg_Loss=loss_func(pred_labels,real_labels.float())#Seg Loss
        Seg_Loss.backward(retain_graph=True)
        fake_pair=torch.cat((real_imgs,pred_labels),dim=1)
        gd_fake=D(fake_pair)
        gd_fake=gd_fake.mean()
        gd_fake.backward(moneg[0])

        optimizer_G.step()
    print("epoch[%d/%d] W:%f segloss%f"%(epoch,args.max_epochs,Wasserstein_D,Seg_Loss))



G.eval()
D.eval()
args.exp = 'TU_' + args.dataset + str(args.img_size)
snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_classes' + str(args.n_classes)
os.makedirs(snapshot_path ,exist_ok=True)
G_save_mode_path=os.path.join(snapshot_path,'G'+'.pth')
D_save_mode_path=os.path.join(snapshot_path,'D'+'.pth')


logging.info("{} test iterations per epoch".format(len(valloader)))
metric_list = np.zeros([len(valloader),10])
metric_out = np.zeros([len(valloader), 10])
for i_val,sample_batch in enumerate(valloader):
    real_imgs = sample_batch['image'].unsqueeze(1).cuda()
    real_labels =sample_batch['label'].unsqueeze(1).cuda()
    name=sample_batch['name']
    logits = G(real_imgs)
    pred_labels = torch.sigmoid(logits)
    pred_mask = (pred_labels > 0.5).long()
    metric_i = calculate_metric_percase(pred_mask.detach().cpu(), real_labels.detach().cpu())
    #valloss = loss_func(outputs, real_labels)
    img_crf = np.array(real_imgs.detach().cpu())
    d = DenseCRF2D(args.img_size, args.img_size, 2)
    u = unary_from_labels(pred_mask.detach().cpu(), 2, gt_prob=0.9, zero_unsure=False)
    d.setUnaryEnergy(u)
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(1,), img=img_crf.squeeze(), chdim=-1)
    d.addPairwiseEnergy(pairwise_energy, compat=6)
    Q = d.inference(12)
    pre_crfs = np.argmax(Q, axis=0).reshape((args.img_size, args.img_size))
    crfs = torch.from_numpy(pre_crfs)

    metric_crf = calculate_metric_percase(crfs, real_labels.detach().cpu())

    name = name[0]
    os.makedirs('./out/img', exist_ok=True)
    os.makedirs('./out/mask', exist_ok=True)
    os.makedirs('./out/out', exist_ok=True)
    os.makedirs('./out/out_crf', exist_ok=True)
    utils.save_image(real_imgs.squeeze(), './out/img/{}'.format(name), normalize=True)
    utils.save_image(real_labels.squeeze().type(torch.float), './out/mask/{}'.format(name))
    utils.save_image(pred_mask.squeeze().type(torch.float), './out/out/{}'.format(name))
    utils.save_image(crfs.squeeze().type(torch.float), './out/out_crf/{}'.format(name))
    c = metric_crf
    metric_list[i_val][:] = np.asarray(c)
    metric_out[i_val][:] = np.asarray(metric_i)
    logging.info('origin-idx %d case %s mean_pod %f mean_far %f' % (i_val, name, metric_i[1], metric_i[2]))
    logging.info('CRF-idx %d case %s mean_pod %f mean_far %f' % (i_val, name, metric_crf[1], metric_crf[2]))


metric_out=np.mean(metric_out,axis=0)
metric_list =np.mean(metric_list,axis=0)
performance = metric_list
print('\n')
print(str(args))
print('\n')
print('epochs:%d lr:%F'%(args.max_epochs,args.base_lr))
print('Testing performance in best val model(origin): mean_pod : %f mean_far : %f mean_precision : %f mean_F1 : %f mean_CSI : %f mean_miou : %f' % (
        metric_out[1], metric_out[2], metric_out[3], metric_out[6], metric_out[4], metric_out[-1]))
print('Testing performance in best val model(CRF): mean_pod : %f mean_far : %f mean_precision : %f mean_F1 : %f mean_CSI : %f mean_miou : %f' % (
        performance[1],performance[2],performance[3],performance[6],performance[4],performance[-1]))

torch.save(G.state_dict(), G_save_mode_path)
torch.save(D.state_dict(), D_save_mode_path)
