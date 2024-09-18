from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 256
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 64
config.lr = 0.1
config.verbose = 200
config.dali = False
config.save_all_states = True

config.rec = r"/root/arcface_torch/data/112"
config.num_classes = 160
config.num_image = 3200
config.num_epoch = 40
config.warmup_epoch = 0
config.val_targets = ['val']
