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
    n_epoch = 10000
    learning_rate = 1e-2
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 100
    num_layers = 3
    seq_length = 12
    log_dir = '/tmp/Generate_text'
    gen_num = 20 # how many chars to generate


class DataGenerator():

    def __init__(self, datafiles, args):
        self.seq_length = args.seq_length
        self.batch_size = args.batch_size
        # 诗集
        self.poetrys = []

        with open(datafiles, "r", encoding='utf-8', ) as f:
            for line in f:
                try:
                    title, content = line.strip().split(':')
                    content = content.replace(' ', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                        continue
                    if len(content) < 5 or len(content) > 79:
                        continue
                    content = '[' + content + ']'
                    self.poetrys.append(content)

                except Exception as e:
                    pass
        all_words = []
        for poetry in self.poetrys:
            for word in poetry:
             all_words += word
        #self.poetrys = sorted(self.poetrys,key=lambda  line:len(line))
        # vocabulary
        self.total_lines = len(self.poetrys)  # total data length
        self.words = list(set(all_words))
        self.words.sort()
        self.vocab_size = len(self.words)  # vocabulary size
        print('Vocabulary Size: ', self.vocab_size)
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}

        self.poetry_vectors = []
        poetrys = self.poetrys
        for poetry in self.poetrys:
            poetry = [self.char2id(c) for c in poetry ]
            self.poetry_vectors.append(poetry)
        self.current_postion = 0

    def next_batch(self,batch_size):
        if self.current_postion + batch_size > self.total_lines:
            self.poetry_vectors = shuffle(self.poetry_vectors)
            self.current_postion = 0
        x_batch = self.poetry_vectors[self.current_postion:self.current_postion+batch_size][:-1]
        y_batch = self.poetry_vectors[self.current_postion:self.current_postion+batch_size][:-1]
        self.current_postion += batch_size
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
                tf.int32, [args.batch_size, args.seq_length],name='x') # 一次读入20个字的长度,
            self.target_data = tf.placeholder(
                tf.int32, [args.batch_size, args.seq_length],name='y') #目标值也是读入20个字的长度

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
            self.cost = tf.reduce_sum(loss) / args.batch_size
            tf.summary.scalar('loss', self.cost)

        with tf.name_scope('optimize'):
            self.lr = args.learning_rate
            optimizer = tf.train.AdamOptimizer(self.lr)

            self.train_op = optimizer.minimize(self.cost)
            self.merged_op = tf.summary.merge_all()



def train(data, model, args):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)


        max_iter = args.n_epoch
        for i in range(max_iter):
            x_batch, y_batch = data.next_batch(args.batch_size)
            train_loss, summary, _ = sess.run([model.cost, model.merged_op, model.train_op],
                                                 feed_dict={model.input_data:x_batch,model.target_data:y_batch})

            if i % 10 == 0:
                writer.add_summary(summary, global_step=i)
                print('Step:{}/{}, training_loss:{:4f}'.format(i,
                                                               max_iter, train_loss))
            if i % 2000 == 0 or (i + 1) == max_iter:
                saver.save(sess, os.path.join(
                    args.log_dir, 'lyrics_model.ckpt'), global_step=i)


def sample(data, model, args):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.latest_checkpoint(args.log_dir)
        print(ckpt)
        saver.restore(sess, ckpt)

        # initial phrase to warm RNN
        prime = u'洪涛经变野'
        state = sess.run(model.cell.zero_state(1, tf.float32))

        for word in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = data.char2id(word)
            feed = {model.input_data: x, model.initial_state: state}
            state = sess.run(model.last_state, feed)

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
        print(msg)
        sys.exit(1)
