from bilm.model.model import dump_bilm_embeddings
import os
import h5py


raw_context = [
'华生园是集食品加工业、旅游服务业、连锁餐饮业、百货商贸业的大型现代化企业，其生产的产品是“中国名点”、“中国名饼”、中国“知名月饼”',
'中央工厂大厅内的咖啡厅、侧边的儿童体验中心即蛋糕“DIY”制作中心和廊道的卡通塑像，营造了欢乐的儿童乐园——金色蛋糕梦幻王国；二、三、四层楼分别是中央工厂的干货、面包、蛋糕生产车间，其产品有面包、蛋糕、生日蛋糕、法式蛋糕、西点、干货等六大类500多个品种的糕点食品'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['坚强']
]

dataset_file = 'dataset_file.txt'
with open(dataset_file, 'w') as fout:
    for sentence in tokenized_context + tokenized_question:
        fout.write(' '.join(sentence) + '\n')

datadir = '.'
vocab_file = os.path.join(datadir, 'vocab/cn_vocab')
options_file = os.path.join(datadir, 'bilm/save_dir/options.json')
weight_file = os.path.join(datadir, 'bilm/save_dir/weights.hdf5')

embedding_file = 'elmo_embeddings.hdf5'
dump_bilm_embeddings(vocab_file, dataset_file, options_file,
                     weight_file, embedding_file)

with h5py.File(embedding_file, 'r') as fin:
    second_sentence_embeddings = fin['1'][...]
    print(second_sentence_embeddings.shape)