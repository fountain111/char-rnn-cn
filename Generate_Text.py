import os
import sys
import time
import collections
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import pandas as pd
#from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import rnn_cell, seq2seq

class HParam():

    batch_size = 128
    n_epoch = 1000000
    learning_rate = 0.02
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 128
    num_layers = 3
    seq_length = 50
    log_dir = '/tmp/Generate_text'
    gen_num = 100 # how many chars to generate


class DataGenerator():

    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size

        with open(datafiles, "r", encoding='utf-8', ) as f:
            self.poetrys = f.read()

        all_words = []
        self.poetry_vectors = []

        for poetry in self.poetrys:
             all_words += poetry
        # vocabulary
        self.total_words = len(self.poetrys)  # total data length
        self.words = list(set(all_words))
        self.words.sort()
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        for poetry in self.poetrys:
            self.poetry_vectors.append(self.char2id(poetry))

        self.current_postion = 0

    def next_batch(self,batch_size,seq_len):
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            if self.current_postion + seq_len + 1 > self.total_words:
                self.current_postion = 0
            x_= self.poetrys[self.current_postion:self.current_postion + seq_len]
            y_ = self.poetrys[self.current_postion+1:self.current_postion + seq_len+1]
            x_ = [self.char2id(c) for c in x_]
            y_ = [self.char2id(c) for c in y_]
            x_batch.append(x_)
            y_batch.append(y_)
            self.current_postion += seq_len

        return x_batch,y_batch








    def char2id(self, c):
        return self.char2id_dict[c]

    def id2char(self, id):
        return self.id2char_dict[id]




class Model():
    """
    The core recurrent neural network model.
    """

    def __init__(self, args, data, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        with tf.name_scope('inputs'):
            self.input_data = tf.placeholder(
                tf.int32, [args.batch_size,args.seq_length],name='x') # 一次读入20个字的长度,
            self.target_data = tf.placeholder(
                tf.int32, [args.batch_size,args.seq_length],name='y') #目标值也是读入20个字的长度

        with tf.name_scope('model'):
            self.cell = rnn_cell.BasicLSTMCell(args.state_size)  #应该是100个unit in the cell
            self.cell = rnn_cell.MultiRNNCell([self.cell] * args.num_layers)   #应该是3层,每层100个CELL的意思
            self.initial_state = self.cell.zero_state(
                args.batch_size, tf.float32)
            with tf.variable_scope('rnnlm'):
                w = tf.get_variable(
                    'softmax_w', [args.state_size, data.vocab_size])
                b = tf.get_variable('softmax_b', [data.vocab_size])
               # with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        'embedding', [data.vocab_size, args.state_size])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            outputs, last_state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(outputs, [-1, args.state_size])

            self.logits = tf.matmul(output, w) + b
            self.probs = tf.nn.softmax(self.logits)
            self.last_state = last_state

            targets = tf.reshape(self.target_data, [-1])
            loss = seq2seq.sequence_loss_by_example([self.logits],
                                                    [targets],
                                                    [tf.ones_like(targets, dtype=tf.float32)]) #使target的概率值最大
            self.cost = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = tf.placeholder(tf.float32, [])

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
            self.merged_op =  tf.summary.merge_all()


def train(data, model, args):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)


        max_iter = args.n_epoch
        for i in range(max_iter):
            learning_rate = args.learning_rate * \
                            (args.decay_rate ** (i // args.decay_steps))
            x_batch, y_batch = data.next_batch(args.batch_size,args.seq_length)
            train_loss, summary, _ = sess.run([model.cost, model.merged_op, model.train_op],
                                                 feed_dict={model.input_data:x_batch,model.target_data:y_batch,model.lr: learning_rate})

            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               max_iter, train_loss))

            if i % 20000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(
                    args.log_dir, 'poe_model.ckpt'), global_step=i)

def to_word(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    return sample

def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        print(ckpt)
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'放'
        state = sess.run(model.cell.zero_state(1, tf.float32))


        word = prime[-1]
        lyrics = prime
        for i in range(args.gen_num):
            x = np.zeros([1, 1])
            x[0, 0] = data.char2id(word)
            feed_dict = {model.input_data: x, model.initial_state: state}
            probs, state = sess.run([model.probs, model.last_state], feed_dict)
            p = probs[0]
            word = data.id2char(np.argmax(p))
            print(word, end='')
            sys.stdout.flush()
            time.sleep(0.05)
            lyrics += word
        return lyrics


def main(infer):
    #infer = True
    args = HParam()
    data = DataGenerator('poetry.txt', args)
    model = Model(args, data, infer=infer)
    #batcx,batchy = data.next_batch()
    #for i in range(len(batcx[0])):
        #print(data.id2char(batcx[0][i]))
        #print("y")
     #   print(data.id2char(batchy[0][i]))


    run_fn = sample if infer else train

    run_fn(data, model, args)


if __name__ == '__main__':
    msg = """
    Usage:
    Training:
        python3 gen_lyrics.py 0
    Sampling:
        python3 gen_lyrics.py 1
    """
    main(0)

    if len(sys.argv) == 2:
        infer = int(sys.argv[-1])
        print('--Sampling--' if infer else '--Training--')
        main(infer)
    else:
        #print(msg)
        sys.exit(1)
