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
config.batch_size = 32
config.lr = 0.005
config.verbose = 10
config.dali = False
config.save_all_states = True
config.using_wandb = False

config.rec =  r"/root/autodl-tmp/test/data/vec"
config.num_classes = 41556
config.num_image = 4155
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['val']
config.val_rec = r"/root/autodl-tmp/test/data/vec" 
