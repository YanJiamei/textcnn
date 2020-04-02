# coding=utf-8
from __future__ import print_function
import pandas as pd
from collections import Counter
import json
import gensim
import numpy as np
import tensorflow as tf
import os
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='',
    help='The directory where data, wv_embedding, json and stop_word stored.'
)
parser.add_argument(
    '--embedding_dir', type=str, default='',
    help=('The directory where the embedding word vectors stored.')
)
parser.add_argument(
    '--model_dir', type=str, default='',
    help='The directory where the model will be stored.'
)
parser.add_argument(
    '--num_epochs', type=int, default=10,
    help='The number of training epochs.'
)
parser.add_argument(
    '--batch_size', type=int, default=64,
    help=('Batch size for training and evaluation.')
)


class Dataset():
    def __init__(self, sequence_len, stop_word_path, wv_path):
        self.sequence_len = sequence_len
        # self.data_path = data_path
        self.stop_word_path = stop_word_path
        self.wv_path = wv_path

        # 读取的停用词
        self.stop_word_dict = dict()

        self.vocab2id = []
        self.label2id = []

        self.embedding_array = None

    def read_data(self, data_path):
        print('reading data from ', data_path, '...')
        df = pd.read_csv(data_path)
        # 读取文本，按空格分词
        contents = df['content'].tolist()
        contents = [line.strip().split() for line in contents]
        # 读取对应label
        labels = df['label'].tolist()
        return contents, labels

    def generate_vocab(self, contents, stop_word_dict, wv_path):
        # 去除文本中的停用词
        # 先整理所有出现的词和次数
        print('generating vocab...')
        all_words = [word for content in contents for word in content]  # 句子的平均长度 len(all_words)/5000 = 477.404
        # 去掉停用词，去除低频词, 建立词汇表vocab
        sub_words = [word for word in all_words if word not in stop_word_dict]  # len(sub_words)/5000 = 255.032
        word_count = Counter(sub_words).most_common()
        vocab = [item[0] for item in word_count if item[1] >= 5]

        # 加载embeddings, 挑出需要的word_vectors
        print('\tloading word vectors...')
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=True)
        # 为vocab添加'UNK'（没有出现过的词）和'PAD'（用于句子对齐）,embedding也相应添加
        vocab.insert(0, 'UNK')
        vocab.insert(1, 'PAD')
        embedding_matrix = []
        embedding_size = 100
        embedding_matrix.append(np.zeros(embedding_size))  # UNK编为0
        embedding_matrix.append(np.random.randn(embedding_size))  # PAD编为一个随机向量
        for word in vocab[2:]:
            if word in word_vecs.wv.vocab:
                embedding_matrix.append(word_vecs[word])
            else:
                embedding_matrix.append(word_vecs['UNK'])
        self.embedding_array = np.array(embedding_matrix)
        # vocab2id
        print('\tbuilding vocab2id...')
        vocab2id = dict(zip(vocab, range(len(vocab))))
        return vocab2id

    def get_embedding_array(self):
        return self.embedding_array

    def generate_label2id(self, labels):
        """生成 label: id 的字典"""
        # ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        unique_labels = list(set(labels))
        label2id = dict(zip(unique_labels, range(len(unique_labels))))
        return label2id

    def transform_data_2_id(self, contents, labels, sequence_len):
        print('transforming datas and labels to id...')
        # 去除停用词（这里去除和不去除停用词的效果不一样，实验证明：
        # contents = [[word for word in words if word not in self.stop_word_dict] for words in contents]

        # 将文本转换成id
        contentids = [[self.vocab2id.get(item, self.vocab2id['UNK']) for item in content] for content in contents]
        processed_data = []
        # 对齐文本为sequence_len长
        for content in contentids:
            if len(content) >= sequence_len:
                processed_data.append(content[:sequence_len])
            else:
                processed_data.append(content + [self.vocab2id['PAD']] * (sequence_len - len(content)))
        x = processed_data
        # 将label转换成id
        y = [self.label2id[label] for label in labels]
        return np.asarray(x), np.array(y)  # 返回np.array 形式

    def _read_stop_words(self, stop_word_path):
        with open(stop_word_path, 'r', encoding='utf-8') as f:
            stop_word = f.read()
            stop_words_list = stop_word.splitlines()
            stop_word_dict = dict(zip(stop_words_list, range(len(stop_words_list))))
        return stop_word_dict  # 返回字典形式而不是列表形式，因为字典查找比较快

    def process_data_generate_vocab(self, train_dir, val_dir):
        '''初次训练的时候需要根据train val数据生成vocab'''
        # 初始化数据
        contents_train, labels_train = self.read_data(train_dir)
        contents_val, labels_val = self.read_data(val_dir)
        contents, labels = contents_train, labels_train
        contents.extend(contents_val)
        labels.extend(labels_val)

        # 读取停用词
        self.stop_word_dict = self._read_stop_words(self.stop_word_path)

        # 生成vocab2id
        self.vocab2id = self.generate_vocab(contents, self.stop_word_dict, self.wv_path)
        # 生成label2id
        self.label2id = self.generate_label2id(labels)

        # 将文本和label转换成id格式
        train_x, train_y = self.transform_data_2_id(contents_train, labels_train, self.sequence_len)
        val_x, val_y = self.transform_data_2_id(contents_val, labels_val, self.sequence_len)
        return train_x, train_y, val_x, val_y

    def process_data_from_exist_vocab(self, data_dir, vocab2id_dir, label2id_dir):
        '''直接根据已经生成好的vocab来处理数据'''
        '''data_dir 的文本转化成id格式，对应的vocab词典已经提前处理好存在vocab2id_dir'''
        # 读取数据
        contents, labels = self.read_data(data_dir)
        # 读取停用词
        self.stop_word_dict = self._read_stop_words(self.stop_word_path)

        # 读取self.vocab2id, self.label2id
        # 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
        print('loading vocab2id, label2id...')
        with open(vocab2id_dir, "r", encoding="utf-8") as f:
            self.vocab2id = json.load(f)
        with open(label2id_dir, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)

        # 将文本和label转换成id格式
        x, y = self.transform_data_2_id(contents, labels, self.sequence_len)
        return x, y

    def save_data2id_json(self, json_path):
        print('saving vocab2id to ', json_path, '...')
        with open(json_path + "/vocab2id.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab2id, f)
        print('saving label2id...')
        with open(json_path + "/label2id.json", "w", encoding="utf-8") as f:
            json.dump(self.label2id, f)

    def save_embedding_array(self, embedding_path):
        print('saving embedding array to ', embedding_path, '...')
        np.save(embedding_path + '/embedding_array.npy', self.embedding_array)


# 输入管道 iterator， 提供生成迭代器
def input_fn(features, labels, training=True, batch_size=128, num_epochs=1):
    # tf.data.Dataset 预处理数据
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # dataset = dataset.shuffle(labels.shape[0])
    if training:
        dataset = dataset.shuffle(10000).repeat(num_epochs)  # 送1000入管道
    #     return dataset
    return dataset.batch(batch_size)


def txt_cnn_model(features, labels, mode, params):
    embedding_wv = tf.convert_to_tensor(params['embedding_array'], dtype=tf.float32, name='word2vec')
    with tf.name_scope('embedding'):
        # 利用预训练词向量初始化嵌入矩阵
        embedding = tf.Variable(embedding_wv, name='embedding')
        embedded = tf.nn.embedding_lookup(embedding, features)
        embedded_expand = tf.expand_dims(embedded, -1)
        # 多个卷积核尺寸，然后合并，并不是多层卷积
        concat_outputs = []
        for i, filter_size in enumerate(params['filter_sizes']):
            with tf.variable_scope('conv-maxpool-%s' % filter_size):
                # 卷积层，卷积核尺寸为filter_size * embedding_dim, 卷积核一共num_fileters个
                # 初始化权重矩阵和偏置 W和b
                filter_shape = [filter_size, params['embedding_dim'], 1, params['num_filters']]
                # filter W
                W = tf.get_variable('W', filter_shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
                # bias b
                b = tf.get_variable('b', [params['num_filters'], ], initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(
                    input=embedded_expand,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',  # 不进行padding
                    name='conv'
                )

                # 激活层
                activated = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # 池化层 max_pooling, textCNN卷积之后height = sequence_len - filter_size, width=1, 池化为height=1, width=1
                pooled = tf.nn.max_pool(
                    activated,
                    ksize=[1, params['seq_length'] - filter_size + 1, 1, 1],  # [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pooling'
                )
                concat_outputs.append(pooled)

        # 将多个channel数据合并
        concat_by_channels = tf.concat(concat_outputs, 3)  # height,width,channel 按照channel
        # reshape成二维，输入到全连接层
        len_fc_input = params['num_filters'] * len(params['filter_sizes'])
        fc_input = tf.reshape(concat_by_channels, [-1, len_fc_input])

        # dropout
        with tf.name_scope('dropout'):
            fc_input_dropped = tf.nn.dropout(fc_input, params['dropout_keep_prob'])

        # 全连接层，输出，不论是什么模式都要输出
        with tf.name_scope('fc'):
            logits = tf.layers.dense(fc_input_dropped, params['num_classes'], name='fc')
            y_pred_cls = tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int32)  # 预测类别

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class_ids': y_pred_cls[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # train和eval模式都要计算loss、acc
        with tf.name_scope('loss'):
            # 损失函数、交叉熵, sparse_softmax输入不是one_hot形式的label
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", loss)  # estimator will automatically write it to tensorboard for you
        with tf.name_scope('accuracy'):
            # 准确率
            #             correct_pred = tf.equal(y_pred_cls, labels)
            #             acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = tf.metrics.accuracy(labels=labels,
                                           predictions=y_pred_cls,
                                           name='acc_op')
            metrics = {'accuracy': accuracy}
            # Create a tensor named train_accuracy for logging purposes.
            tf.identity(accuracy[1], name='train_accuracy')
            tf.summary.scalar('train_accuracy', accuracy[1])
        # 此时可以判断如果是评估eval可以进行数据返回
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        # 在训练模式下需要进行更新优化
        assert mode == tf.estimator.ModeKeys.TRAIN
        # 优化损失函数
        optim = tf.train.AdamOptimizer(params['learning_rate'])
        train_op = optim.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def save_tfrecords(data, label, desfile):
    with tf.python_io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature={
                    "data": tf.train.Feature(bytes_list=tf.train.Int64List(value=[data[i]])),
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def main(unused_argv):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    train_dir = os.path.join(FLAGS.data_dir, 'cnews/cnews.trainCut.csv')
    val_dir = os.path.join(FLAGS.data_dir, 'cnews/cnews.valCut.csv')
    # 停用词和word2vec词典
    stop_word_path = os.path.join(FLAGS.data_dir, 'stopWords.txt')
    wv_path = os.path.join(FLAGS.data_dir, 'embed_weights/embedding_unk.bin')

    # vocab2id label2id embedding_array_dir
    vocab_json_path = os.path.join(FLAGS.data_dir, 'embed_weights/vocab2id.json')
    label_json_path = os.path.join(FLAGS.data_dir, 'embed_weights/label2id.json')
    # embedding_array_dir = os.path.join(FLAGS.data_dir, 'embed_weights/embedding_array.npy')


    print('Loading training and validation data...')
    seq_length = 256  # 平均长度
    dataset = Dataset(seq_length, stop_word_path, wv_path)
    #HDFS不兼容np.load 使用相对路径
    embedding_array = np.load(FLAGS.embedding_dir)

    params = {  # 模型的额外参数，会传递给模型参数的params
        'embedding_dim': 100,  # 词向量维度
        'seq_length': 256,  # 序列长度
        'num_classes': 10,  # 类别数
        'num_filters': 128,  # 卷积核数目
        'filter_sizes': [3, 4, 5],  # 卷积核尺寸

        'dropout_keep_prob': 0.5,  # dropout保留比例
        'learning_rate': 1e-3,  # 学习率

        'batch_size': FLAGS.batch_size,  # 每批训练大小
        'num_epochs': FLAGS.num_epochs,  # 总迭代轮次

        'print_per_batch': 100,  # 每多少轮输出一次val结果, 并存模型进best validation
        'save_per_batch': 10,  # 每多少轮存入tensorboard
        'embedding_array': embedding_array
    }

    model_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                          save_checkpoints_steps=100,  # 每200步保存一次
                                          keep_checkpoint_max=3)  # 保留最新的3个checkpoints
    model = tf.estimator.Estimator(
        config=model_config,
        model_fn=txt_cnn_model,  # 制定本模型的模型参数
        params=params)

    print('Starting training...')
    x_train, y_train = dataset.process_data_from_exist_vocab(train_dir, vocab_json_path, label_json_path)
    model.train(input_fn=lambda: input_fn(x_train, y_train, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs))
    print('Start evaluating...')
    x_val, y_val = dataset.process_data_from_exist_vocab(val_dir, vocab_json_path, label_json_path)
    model.evaluate(input_fn=lambda: input_fn(x_val, y_val, batch_size=FLAGS.batch_size, training=False))


# 数据集
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
