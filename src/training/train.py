import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from open_clip.loss import ClipLossPlusReconstruction, ClipLoss


try:
    import wandb
except ImportError:
    wandb = None

from open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
import torchvision
from torchvision import transforms
import srt.srt.utils.visualize as vis
from srt.srt.utils import nerf


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def train_one_epoch(model, teacher_encoder, data, msn_loader, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    msn = False
    if args.msn_path != None:
        msn = True

    model.train()
    if msn:
        loss = ClipLossPlusReconstruction(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
            recon_lambda = args.recon_lambda)
    else:
        loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))


    msn_iterator = iter(msn_loader)

    loss_m = AverageMeter()
    recon_loss = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images, texts = batch

        data = next(msn_iterator)
        msn_images = data['input_images']
        input_camera_pos = data['input_camera_pos']
        input_rays = data['input_rays']
        target_pixels = data['target_pixels']
        target_camera_pos = data['target_camera_pos']
        target_rays = data['target_rays']






        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)

        
        if msn == True:
            msn_images = msn_images.to(device=device, non_blocking=True)
            input_camera_pos = input_camera_pos.to(device=device, non_blocking=True)
            input_rays = input_rays.to(device=device, non_blocking=True)
            target_pixels = target_pixels.to(device=device, non_blocking=True)
            target_camera_pos = target_camera_pos.to(device=device, non_blocking=True)
            target_rays = target_rays.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()


        with autocast():
            with torch.no_grad:
                z_teacher = teacher_encoder(msn_images, input_camera_pos, input_rays)
                z_teahcer = z_teacher.flatten(1,2)

            image_features, text_features, logit_scale = model(images, texts)
            if msn == True:
                z = model(msn_images, None, input_camera_pos, input_rays)
                z = z.flatten(1,2)
                total_loss, r_loss = loss(image_features, text_features, z_teacher, z, logit_scale)
            else:
                total_loss = loss(image_features, text_features,logit_scale)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.norm_gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if is_master(args) and (i % 100 == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            msn_batch_size = msn_images.shape[0]
            num_samples = batch_count * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)
            recon_loss.update(r_loss.item(),msn_batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss_recon: {recon_loss.val:#.5g} ({recon_loss.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "loss_recon": recon_loss.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size*args.world_size / batch_time_m.val,
                "scale":  logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    wandb.log({name: val, 'step': step})

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)

    
    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples
        

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def evaluate_msn(model, SRTdecoder, msn_loader, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    SRTdecoder.eval()


    autocast = get_autocast(args.precision)

    
    if (args.msn_val_frequency and ((epoch % args.msn_val_frequency) == 0 or epoch == args.epochs)):
        dataloader = msn_loader
        num_samples = 0
        samples_per_val = 1000
        cumulative_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                #msn_images, input_camera_pos, input_rays, target_pixels, target_camera_pos, target_rays = batch
                #data = next(msn_iterator)
                msn_images = data['input_images']
                input_camera_pos = data['input_camera_pos']
                input_rays = data['input_rays']
                target_pixels = data['target_pixels']
                target_camera_pos = data['target_camera_pos']
                target_rays = data['target_rays']


                msn_images = msn_images.to(device=device, non_blocking=True).flatten(0,1)
                input_camera_pos = input_camera_pos.to(device=device, non_blocking=True)
                input_rays = input_rays.to(device=device, non_blocking=True)
                target_pixels = target_pixels.to(device=device, non_blocking=True)
                target_camera_pos = target_camera_pos.to(device=device, non_blocking=True)
                target_rays = target_rays.to(device=device, non_blocking=True)

                with autocast():
                    z = model(msn_images, None, input_camera_pos, input_rays)
                    pred_pixels, extras = SRTdecoder(z, target_camera_pos, target_rays)


                    batch_size = msn_images.shape[0]
                    total_loss = ((pred_pixels - target_pixels)**2).mean((1, 2))
                    total_loss = total_loss.mean(0)                   

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"MSN Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t")

            loss = cumulative_loss / num_samples
            metrics.update(
                {"msn_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    if not metrics:
        return metrics

    logging.info(
        f"MSN Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        for name, val in metrics.items():
            wandb.log({f"val/{name}": val, 'epoch': epoch})

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def render_image(z, srtdecoder, args, camera_pos, rays):
    inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    batch_size, height, width = rays.shape[:3]
    rays = rays.flatten(1, 2)
    camera_pos = camera_pos.unsqueeze(1).repeat(1, rays.shape[1], 1)

    max_num_rays = 8192 * \
                args.msn_batch_size // (rays.shape[0])
    num_rays = rays.shape[1]
    img = torch.zeros_like(rays)
    all_extras = []
    for i in range(0, num_rays, max_num_rays):
        img[:, i:i+max_num_rays], extras = srtdecoder(
                z=z, x=camera_pos[:, i:i+max_num_rays], rays=rays[:, i:i+max_num_rays])
        all_extras.append(extras)

    agg_extras = {}
    for key in all_extras[0]:
        agg_extras[key] = torch.cat([extras[key] for extras in all_extras], 1)
        agg_extras[key] = agg_extras[key].view(batch_size, height, width, -1)

    img = img.view(img.shape[0], height, width, 3)
    img = torch.permute(img,(0,3,1,2))
    img = inv_trans(img)
    img = torch.permute(img,(0,2,3,1))


    return img, agg_extras


def visualize(model, srtdecoder, args, data, epoch, mode='val'):
    model.eval()
    autocast = get_autocast(args.precision)
    inv_trans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711 ]),
                                transforms.Normalize(mean = [ -0.48145466, -0.4578275, -0.40821073 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    with torch.no_grad():
        device = torch.device(args.device)
        input_images = data.get('input_images').to(device)
        input_camera_pos = data.get('input_camera_pos').to(device)
        input_rays = data.get('input_rays').to(device)

        camera_pos_base = input_camera_pos[:, 0]
        input_rays_base = input_rays[:, 0]

        if 'transform' in data:
            # If the data is transformed in some different coordinate system, where
            # rotating around the z axis doesn't make sense, we first undo this transform,
            # then rotate, and then reapply it.
                
            transform = data['transform'].to(device)
            inv_transform = torch.inverse(transform)
            camera_pos_base = nerf.transform_points_torch(camera_pos_base, inv_transform)
            input_rays_base = nerf.transform_points_torch(input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
        else:
            transform = None

        input_images_np = np.transpose(inv_trans(input_images).cpu().numpy(), (0, 1, 3, 4, 2))

        with autocast():
            z = model(input_images.flatten(0,1), None,  input_camera_pos, input_rays)

        batch_size, num_input_images, height, width, _ = input_rays.shape

        num_angles = 6

        columns = []
        for i in range(num_input_images):
            header = 'input' if num_input_images == 1 else f'input {i+1}'
            columns.append((header, input_images_np[:, i], 'image'))

        all_extras = []
        for i in range(num_angles):
            angle = i * (2 * math.pi / num_angles)
            angle_deg = (i * 360) // num_angles

            camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos_base, angle)
            rays_rot = nerf.rotate_around_z_axis_torch(input_rays_base, angle)

            if transform is not None:
                camera_pos_rot = nerf.transform_points_torch(camera_pos_rot, transform)
                rays_rot = nerf.transform_points_torch(rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)

            img, extras = render_image(z, srtdecoder, args, camera_pos_rot, rays_rot)
            all_extras.append(extras)
            columns.append((f'render {angle_deg}°', img.cpu().numpy(), 'image'))


        output_img_path = os.path.join(args.checkpoint_path, f'renders-{mode}-{epoch}')
        vis.draw_visualization_grid(columns, output_img_path)
