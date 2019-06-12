import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from octconv_net import *
from gluoncv.utils import makedirs, LRSequential, LRScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Train a Octave based Model.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epochs at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--alpha', type=float, default=0,
                    help='model param.')
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
# parser.add_argument('--use_se', action='store_true',
#                     help='use SE layers or not in resnext. default is false.')
parser.add_argument('--mixup', action='store_true',
                    help='whether train the model with mix-up. default is false.')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument('--mixup-off-epoch', type=int, default=0,
                    help='how many last epochs to train without mixup, default is 0.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                    help='name of training log file')
opt = parser.parse_args()

filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)
rec_train = os.path.join(opt.data_dir, 'train.rec')
rec_train_idx = os.path.join(opt.data_dir, 'train.idx')
rec_val = os.path.join(opt.data_dir, 'val.rec')
rec_val_idx = os.path.join(opt.data_dir, 'val.idx')

batch_size = opt.batch_size
classes = 1000
num_training_samples = 1281167

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
num_batches = num_training_samples // batch_size

lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.num_epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])

model_name = opt.model
optimizer = 'nag'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

net = get_model(model_name, opt.alpha) if opt.alpha > 0 else get_model(model_name)
net.cast(opt.dtype)


def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=num_workers,
        shuffle=True,
        batch_size=batch_size,

        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
        rand_mirror=True,
        random_resized_crop=True,
        max_aspect_ratio=4. / 3.,
        min_aspect_ratio=3. / 4.,
        max_random_area=1,
        min_random_area=0.08,
        brightness=jitter_param,
        saturation=jitter_param,
        contrast=jitter_param,
        pca_noise=lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=False,
        batch_size=batch_size,

        resize=resize,
        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2],
    )
    return train_data, val_data, batch_fn


train_data, val_data, batch_fn = get_data_rec(rec_train, rec_train_idx,
                                              rec_val, rec_val_idx,
                                              batch_size, num_workers)

if opt.mixup:
    train_metric = mx.metric.RMSE()
else:
    train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
save_frequency = opt.save_frequency

if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0


def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        res.append(lam * y1 + (1 - lam) * y2)
    return res


def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
        smoothed.append(res)
    return smoothed


def test(ctx, val_data):
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1 - top1, 1 - top5)


def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    if opt.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    if opt.label_smoothing or opt.mixup:
        sparse_label_loss = False
    else:
        sparse_label_loss = True

    L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

    best_val_score = 1

    for epoch in range(opt.num_epochs):
        tic = time.time()

        train_data.reset()
        train_metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)

            if opt.mixup:
                lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                    lam = 1
                data = [lam * X + (1 - lam) * X[::-1] for X in data]

                if opt.label_smoothing:
                    eta = 0.1
                else:
                    eta = 0.0
                label = mixup_transform(label, classes, lam, eta)

            elif opt.label_smoothing:
                hard_label = label
                label = smooth(label, classes)

            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)

            if opt.mixup:
                output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                  for out in outputs]
                train_metric.update(label, output_softmax)
            else:
                if opt.label_smoothing:
                    train_metric.update(hard_label, outputs)
                else:
                    train_metric.update(label, outputs)

            if opt.log_interval and not (i + 1) % opt.log_interval:
                train_metric_name, train_metric_score = train_metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                    epoch, i, batch_size * opt.log_interval / (time.time() - btic),
                    train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        train_metric_name, train_metric_score = train_metric.get()
        throughput = int(batch_size * i / (time.time() - tic))

        err_top1_val, err_top5_val = test(ctx, val_data)

        logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
        logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f' % (epoch, err_top1_val, err_top5_val))

        if err_top1_val < best_val_score:
            best_val_score = err_top1_val
            net.save_parameters(
                '%s/%.4f-imagenet-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
            trainer.save_states(
                '%s/%.4f-imagenet-%s-%d-best.states' % (save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, epoch))
            trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, opt.num_epochs - 1))
        trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, opt.num_epochs - 1))


if __name__ == '__main__':
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    train(context)
