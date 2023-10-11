# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import shutil
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
from utils.utils import distributed_all_gather, AverageMeter, dice
import torch.utils.data.distributed
from monai.data import decollate_batch


def train_epoch(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss_l1 = AverageMeter()
    run_loss_l2 = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data['image'], batch_data['label']
        data, target = data.cuda(args.local_rank), target.cuda(args.local_rank)
        # for param in model.parameters(): param.grad = None
        optimizer.zero_grad()
        model.zero_grad()
        with autocast(enabled=args.amp):
            logits = model(data, False)
            loss, triplet_loss, dice_loss = loss_func(logits, target, epoch)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_loss_l1.update(triplet_loss.item(), n=args.batch_size)
            run_loss_l2.update(dice_loss.item(), n=args.batch_size)
        if args.local_rank == 0:
            print('Epoch {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                  'loss: {:.4f}'.format(run_loss.avg),
                  'loss_triplet: {:.4f}'.format(run_loss_l1.avg),
                  'loss_dice: {:.4f}'.format(run_loss_l2.avg),
                  'time {:.2f}s'.format(time.time() - start_time))
        start_time = time.time()
    for param in model.parameters() : param.grad = None
    return run_loss.avg

def val_epoch(model,
              loader,
              epoch,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            torch.cuda.empty_cache()
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data['image'], batch_data['label']
            data, target = data.cuda(args.local_rank), target.cuda(args.local_rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                # target = target.cpu()
                logits = logits.cuda(args.local_rank)
    
            if args.distributed:
                val_labels_list = decollate_batch(target)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(logits[:,:args.out_channels])
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                acc_func.reset()
                acc_func(y_pred=val_output_convert, y=val_labels_convert)
                acc, not_nans = acc_func.aggregate()
                acc = acc.cuda(args.local_rank)
            else:
                val_outputs = torch.argmax(logits, dim=1)[0]
                val_labels = target[0, 0, :, :, :]
                acc = 0

                for i in range(1, args.out_channels):
                    organ_Dice = dice(val_outputs == i, val_labels == i)
                    acc += organ_Dice

                acc /= (args.out_channels - 1)
                not_nans = 1

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather([acc, not_nans],
                                                                 out_numpy=True,
                                                                 is_valid=idx < loader.sampler.valid_length)
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc, n=not_nans)

            if args.local_rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print('Val {}/{} {}/{}'.format(epoch, args.max_epochs, idx, len(loader)),
                      'acc', avg_acc,
                      'time {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
    return run_acc.avg

def save_checkpoint(model,
                    epoch,
                    args,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None,
                    loss=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    if loss is not None:
        save_dict['loss'] = loss.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 post_label=None,
                 post_pred=None
                 ):
    writer = None
    if args.logdir is not None and args.local_rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.local_rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.local_rank, time.ctime(), 'Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        epoch_time = time.time()
        train_loss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 loss_func=loss_func,
                                 args=args)
        if args.local_rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        if args.local_rank==0 and writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        if (epoch+1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(model,
                                    val_loader,
                                    epoch=epoch,
                                    acc_func=acc_func,
                                    model_inferer=model_inferer,
                                    args=args,
                                    post_label=post_label,
                                    post_pred=post_pred)

            val_avg_acc = np.mean(val_avg_acc)

            if args.local_rank == 0:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                      'acc', val_avg_acc, 'time {:.2f}s'.format(time.time() - epoch_time))
                if writer is not None:
                    writer.add_scalar('val_acc', val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print('new best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.local_rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args,
                                        best_acc=val_acc_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        loss=loss_func)
            if args.local_rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_acc_max,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                loss=loss_func,
                                filename='model_final.pt')
                if b_new_best:
                    print('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        if scheduler is not None:
            scheduler.step()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max
