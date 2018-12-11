import os
import tensorflow as tf
import json
from bilm.model.util import load_vocab, dump_weights
from bilm.model.training import train
from bilm.data_process.bidirectionallmdataset import BidirectionalLMDataset
from bilm.model.languagemodel import LanguageModel

path = os.path.dirname(os.path.realpath(__file__))
# print(path)
config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bilm')
params_path = os.path.join(config_path, 'config/params.json')
# print(params_path)
#/root/PycharmProjects/elmo/bilm/config/params.json
with open(params_path) as param:
    params_dict = json.load(param)
# print(params_dict)
config = tf.contrib.training.HParams(**params_dict)


#==================测试dump_weights函数=================
# with open(os.path.join(config.save_dir, 'options.json'), 'w') as fout:
#     fout.write(json.dumps(params_dict))
# vocab = load_vocab(config.vocab_file, max_word_length=config.char_cnn['max_characters_per_token'])
# data = BidirectionalLMDataset(config.train_prefix, vocab, test=False,
#                               shuffle_on_load=True)
# train(config, data, n_gpus=config.n_gpus, tf_save_dir=config.save_dir,
#       tf_log_dir=config.save_dir, restart_ckpt_file=None)
save_dir = os.path.join(path, config.save_dir)
dump_weights(save_dir, os.path.join(save_dir, "weights_1.hdf5"))


#====================测试LM模型==================
# model = LanguageModel(config, True)
# print(model.total_loss)


#=============测试train函数的一部分=================
# vocab = load_vocab(config.vocab_file, max_word_length=config.char_cnn['max_characters_per_token'])
# data = BidirectionalLMDataset(config.train_prefix, vocab, test=False,
#                               shuffle_on_load=True)
# train(config, data, n_gpus=config.n_gpus, tf_save_dir=config.save_dir,
#       tf_log_dir=config.save_dir, restart_ckpt_file=None)



#==============测试BidirectionalLMDataset================
# vocab = load_vocab(config.vocab_file, max_word_length=5)
# data = BidirectionalLMDataset(config.train_prefix, vocab, test=False,
#                               shuffle_on_load=True)
# for r in data.iter_batches(config.batch_size, config.unroll_steps):
#     print(r)
#     break


#====================测试load_vocab函数=========
# vocab = load_vocab(config.vocab_file, max_word_length=None)
# print(vocab.size)


#===============list的pop函数==============
# print([1, 2].pop())


#=========测试配置文件
# config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
# params_path = os.path.join(config_path, 'params.json')
# print(params_path)
# with open(params_path) as param:
#     params_dict = json.load(param)
# print(params_dict)
# config = tf.contrib.training.HParams(**params_dict)
# print(config.char_cnn['embedding']['dim'])