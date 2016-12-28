import tensorflow as tf
from tensorflow.python.ops import rnn_cell, seq2seq

class Model():
    def __init__(self,test,arg,data):
        if test: #进入test,非训练
            arg.batch_size = 1 #
            arg.seq_len = 1
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32,[arg.batch_size,arg.seq_len])
            self.y = tf.placeholder(tf.int32,[arg.batch_size,arg.seq_len])

        with tf.name_scope('model'):
            self.cell = rnn_cell.BasicLSTMCell(arg.state_size)
            self.cell = rnn_cell.MultiRNNCell([self.cell] * arg.num_layers)  # 应该是3层,每层100个CELL的意思
            self.initial_state = self.cell.zero_state(arg.batch_size, tf.float32)


            with tf.variable_scope('variable'):
                w = tf.get_variable('softMax_W',[arg.state_size,data.vocab_size])
                b = tf.get_variable('softMax_B',[data.vocab_size])
                emmbeding = tf.get_variable('emmbdeing',[arg.vocab_size,arg.state_size])
                rnn_input = tf.nn.embedding_lookup(emmbeding,self.x)
            output,_ = tf.nn.dynamic_rnn(self.cell,rnn_input,initial_state=self.initial_state)

        with tf.name_scope('loss'):
            output = tf.reshape(output,[-1,arg.state_size])
            self.logits = tf.matmul(output, w) + b
            self.prob   = tf.nn.softmax(logits=self.logits,name='prob')
