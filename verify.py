

import json
import math
import os

from absl import app
from absl import flags
from absl import logging
import model as model_lib
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import numpy as np
import torch
from resnet import get_resnet, get_head

from resnet_tf import BatchNormRelu, Conv2dFixedPadding, IdentityLayer

from pdb import set_trace as pb


FLAGS = flags.FLAGS

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', False,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')


flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 64,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 2,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', False,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 18,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 32,
    'Input image size.')
# https://stackoverflow.com/questions/47895494/how-to-convert-all-layers-of-a-pretrained-keras-model-to-a-different-dtype-from
# https://stackoverflow.com/questions/59400128/warningtensorflowlayer-my-model-is-casting-an-input-tensor-from-dtype-float64
def main(argv):
  # ==========================================
  # verify initilisation
  if False:
    num_classes = 10
    depth = 18
    width = 1
    sk_ratio = 0

    channels_in=512
    num_layers=2
    out_dim=64
    pth_path = 'r18_1x_simclrv2.pth'
    tf_path = '../simclr/tf2/cifar10_models/firsttry_real/ckpt-780'
  else:
    num_classes = 10
    depth = 50
    FLAGS.resnet_depth = 50
    width = 1
    sk_ratio = 0

    channels_in=2048
    num_layers=2
    out_dim=64
    pth_path = 'r50_1x_simclrv2.pth'
    tf_path = '../simclr/tf2/cifar10_models/firsttry_real50/ckpt-1'

  model_pt = get_resnet(depth, width, sk_ratio, cifar_stem=True)
  head_pt = get_head(channels_in, num_layers, out_dim)

  np.random.seed(0)
  imarray = np.random.rand(10,32,32,3)
  imarray_tf = tf.convert_to_tensor(imarray)

  model = model_lib.Model(num_classes)  
  tf_proj, tf_hidden = model(imarray_tf,True)

  if False:
    model_pt.net[1].blocks[0].net.conv1.weight.mean()
    model_pt.net[1].blocks[0].net.conv1.weight.max()
    model_pt.net[1].blocks[0].net.conv1.weight.min()
    model_pt.net[1].blocks[0].net.conv1.weight.std()

    tf.math.reduce_mean(model.layers[0].block_groups[0].layers[0].conv2d_bn_layers[0].conv2d.kernel)
    tf.math.reduce_max(model.layers[0].block_groups[0].layers[0].conv2d_bn_layers[0].conv2d.kernel)
    tf.math.reduce_min(model.layers[0].block_groups[0].layers[0].conv2d_bn_layers[0].conv2d.kernel)
    tf.math.reduce_std(model.layers[0].block_groups[0].layers[0].conv2d_bn_layers[0].conv2d.kernel)

  # ==========================================
  # torch.set_default_tensor_type(torch.DoubleTensor)
  # tf.keras.backend.set_floatx('float64')
  np.random.seed(0)
  imarray = np.random.rand(10,32,32,3)
  imarray_tf = tf.convert_to_tensor(imarray)
  imarray_pt = torch.tensor(imarray).permute(0,3,1,2)
  # imarray_pt.permute(0,2,3,1)
  # model_pt.net[0].conv1(imarray_pt).permute(0,2,3,1)
  # ==========================================
  # depth, width, sk_ratio = name_to_params(args.tf_path)
  # depth = 18
  # width = 1
  # sk_ratio = 0

  # channels_in=512
  # num_layers=2
  # out_dim=64
  # pth_path = 'r18_1x_simclrv2.pth'
  # model_pt = get_resnet(depth, width, sk_ratio, cifar_stem=True)
  # head_pt = get_head(channels_in, num_layers, out_dim)

  model_pt.load_state_dict(torch.load(pth_path)['resnet'])
  head_pt.load_state_dict(torch.load(pth_path)['head'])
  
  model_pt.double()
  head_pt.double()


  # torch.save({'resnet': model.state_dict(), 'head': head.state_dict()}, save_location)
  # ==========================================

  num_classes = 10
  model = model_lib.Model(num_classes)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(tf_path).expect_partial()
  model = checkpoint.model
  
  tf_proj, tf_hidden = model(imarray_tf,True)
  weights = model.get_weights()
  weights = [tf.cast(w, tf.float64) for w in weights]
  tf.keras.backend.set_floatx('float64')
  model = model_lib.Model(num_classes)
  imarray_tf2 = tf.cast(imarray_tf, tf.float64)
  tf_proj, tf_hidden = model(imarray_tf2,True)

  model.set_weights(weights)
  # for m in model.submodules:
  #   if isinstance(m, BatchNormRelu):
  #     m.bn.gamma = tf.cast(m.bn.gamma, tf.float64)
  #     m.bn.moving_variance = tf.cast(m.bn.moving_variance, tf.float64)
  #     m.bn.moving_mean = tf.cast(m.bn.moving_mean, tf.float64)
  #     if m.bn.beta is not None:
  #       m.bn.beta = tf.cast(m.bn.beta, tf.float64)
  #   elif isinstance(m, Conv2dFixedPadding):
  #     m.conv2d.kernel = tf.cast(m.conv2d.kernel, tf.float64)
  #   elif isinstance(m, tf.keras.layers.Conv2D):
  #     m.kernel = tf.cast(m.kernel, tf.float64)
  #   elif isinstance(m, model_lib.LinearLayer):
  #     m.dense.kernel = tf.cast(m.dense.kernel, tf.float64)
  # ==========================================
  imarray_tf2 = tf.cast(imarray_tf, tf.float64)
  tf_proj, tf_hidden = model(imarray_tf2,True)
  # ==========================================
  pt_hidden = model_pt(imarray_pt)
  pt_proj = head_pt(pt_hidden)

  print((np.abs(tf_hidden.numpy() - pt_hidden.detach().numpy()) > 1e-6).sum())
  print((np.abs(tf_proj.numpy() - pt_proj.detach().numpy()) > 1e-6).sum())
  pb()



if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  # For outside compilation of summaries on TPU.
  tf.config.set_soft_device_placement(True)
  # os.environ["TF_DETERMINISTIC_OPS"] = "1"
  tf.random.set_seed(1)
  # tf.data.experimental.enable_debug_mode()
  app.run(main)

