import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

class ProductClassifier():

    def __init__(self, filterSizes, numClasses, learningRate=0.001,
                embeddingSize=128, batchSize=64, fcSize=128, keepProb=0.75,
                resume=1, mode='train'):
        self.filterSizes = filterSizes
        self.numClasses = numClasses
        self.learningRate = learningRate
        self.embeddingSize = embeddingSize
        self.batchSize = batchSize
        self.fcSize = fcSize
        self.numEpoch = 1000
        self.textLength = 48
        self.numFilters = 32
        self.keepProb = keepProb
        with open('./Utility/vocab') as f, open('./Utility/w2idx') as g:
            self.vocab = pickle.load(f)
            self.w2idx = pickle.load(g)
        self.XTrain = np.load('./Dataset/XTrain.npy')
        self.yTrain = np.load('./Dataset/yTrain.npy')
        self.XTest = np.load('./Dataset/XTest.npy')
        self.yTest = np.load('./Dataset/yTest.npy')
        self.lb = preprocessing.LabelBinarizer()
        if mode == 'train':
            classes = self.yTrain
            classes = classes.reshape(-1, 1)
            self.lb.fit(classes)
            with open('lb', 'wb') as f:
                pickle.dump(self.lb, f)
            self.inputDict = {
                "text": tf.placeholder(tf.int32, shape=[None, self.textLength]),
                "classes": tf.placeholder(tf.int32, shape=[None, self.numClasses]),
                "keepProb": tf.placeholder(tf.float32)
            }
        if mode == 'validation':
            self.inputDict = {
                "text": tf.placeholder(tf.int32, shape=[None, self.textLength]),
                "classes": tf.placeholder(tf.int32, shape=[None, self.numClasses])
            }
        self.currentEpoch = 0
        self.currentStep = 0
        self.resume = resume
        if self.resume is 1:
            if os.path.isfile('model/save.npy'):
                self.currentEpoch, self.currentStep = np.load("model/save.npy")
            else:
                print "No Checkpoints Available, Restarting Training.."
        self.Wemb = self.init_weight([len(self.vocab), self.embeddingSize])

    def get_batch(self):
        for batch_idx in range(0, len(self.XTrain), self.batchSize):
            textBatch = self.XTrain[batch_idx:batch_idx + self.batchSize]
            classBatch = self.yTrain[batch_idx:batch_idx + self.batchSize]
            classBatch = self.lb.transform(classBatch)
            yield textBatch, classBatch

    def get_val_batch(self):
        for batch_idx in range(0, len(self.XTest), self.batchSize):
            textBatch = self.XTest[batch_idx:batch_idx + self.batchSize]
            classBatch = self.yTest[batch_idx:batch_idx + self.batchSize]
            classBatch = self.lb.transform(classBatch)
            yield textBatch, classBatch

    def init_weight(self, shape):
        return tf.Variable(tf.random_uniform(shape) * 2 - 1)

    def init_bias(self, shape):
        return tf.Variable(tf.zeros(shape))

    def conv2D_layer(self, inp, kernelShape, biasShape):
        weights = self.init_weight(kernelShape)
        bias = self.init_bias(biasShape)
        conv = tf.nn.conv2d(inp, weights, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.relu(conv + bias)

    def pool_layer(self, inp, kernelShape):
        return tf.nn.max_pool(inp, ksize=kernelShape, strides=[1, 1, 1, 1], padding='VALID')

    def fc_layer(self, inp, inpShape, outShape, activation=False):
        weights = self.init_weight([inpShape, outShape])
        bias = self.init_bias(outShape)
        out = tf.matmul(inp, weights) + bias
        if activation:
            return tf.nn.relu(out)
        return out

    def build_training_graph(self):
        embedding = tf.nn.embedding_lookup(self.Wemb, self.inputDict['text'])
        embedding = tf.expand_dims(embedding, -1)

        pooledOutputs = []
        for filterSize in self.filterSizes:
            filterShape = [filterSize, self.embeddingSize, 1, self.numFilters]
            conv = self.conv2D_layer(embedding, [filterSize, self.embeddingSize, 1, self.numFilters], [self.numFilters])
            pooled = self.pool_layer(conv, [1, self.textLength - filterSize + 1, 1, 1])
            pooledOutputs.append(pooled)
        stackedFeatures = tf.concat(pooledOutputs, axis=3)
        flatStackedFeatures = tf.reshape(stackedFeatures, [-1, len(self.filterSizes)*self.numFilters])
        fc1Features = self.fc_layer(flatStackedFeatures, len(self.filterSizes)*self.numFilters, self.fcSize, True)
        dropFeatures = tf.nn.dropout(fc1Features, self.inputDict['keepProb'])
        fc2Features = self.fc_layer(dropFeatures, self.fcSize, self.numClasses, False)
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2Features, labels=self.inputDict['classes'])
        loss = tf.reduce_mean(crossEntropy)
        return loss, self.inputDict


    def train(self, loss, inputDict):
        self.loss = loss
        self.inputDict = inputDict
        saver = tf.train.Saver(max_to_keep=10)
        globalStep = tf.Variable(self.currentStep, name='globalStep', trainable=False)
        learningRate = tf.train.exponential_decay(
            self.learningRate, globalStep, 50000, 0.95, staircase=True)
        optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("learning_rate", learningRate)
        summaryOp = tf.summary.merge_all()

        print 'Begin Training'
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if self.resume is 1:
                print "Loading Previously Trained Model"
                try:
                    ckptFile = "./model/model.ckpt-" + str(self.currentStep)
                    saver.restore(sess, ckptFile)
                    print "Resuming Training"
                except Exception as e:
                    print str(e).split('\n')[0]
                    print "Checkpoints not found"
                    sys.exit(0)

            writer = tf.summary.FileWriter("model/log_dir/", graph=tf.get_default_graph())
            for epoch in range(self.currentEpoch, self.numEpoch):
                loss = []
                batchIter = self.get_batch()
                for batch in xrange(0, len(self.XTrain), self.batchSize):
                    textBatch, actualClass = batchIter.next()
                    feedDict = {}
                    feedDict[self.inputDict['text']] = textBatch
                    feedDict[self.inputDict['classes']] = actualClass
                    feedDict[self.inputDict['keepProb']] = self.keepProb
                    run = [globalStep, optimizer, self.loss, summaryOp]
                    step, _, curLoss, summary = sess.run(
                        run, feed_dict=feedDict)
                    writer.add_summary(summary, step)
                    if step % 10 == 0:
                        print epoch, ": Global Step:", step, "\tLoss: ", curLoss
                    loss.append(curLoss)
                print
                print "Epoch: ", epoch, "\tAverage Loss: ", np.mean(loss)
                print "\nSaving Model..\n"
                saver.save(sess, "./model/model.ckpt", global_step=globalStep)
                np.save("model/save", (epoch, step))


    def build_validation_graph(self):
        embedding = tf.nn.embedding_lookup(self.Wemb, self.inputDict['text'])
        embedding = tf.expand_dims(embedding, -1)

        pooledOutputs = []
        for filterSize in self.filterSizes:
            filterShape = [filterSize, self.embeddingSize, 1, self.numFilters]
            conv = self.conv2D_layer(embedding, [filterSize, self.embeddingSize, 1, self.numFilters], [self.numFilters])
            pooled = self.pool_layer(conv, [1, self.textLength - filterSize + 1, 1, 1])
            pooledOutputs.append(pooled)
        stackedFeatures = tf.concat(pooledOutputs, axis=3)
        flatStackedFeatures = tf.reshape(stackedFeatures, [-1, len(self.filterSizes)*self.numFilters])
        fc1Features = self.fc_layer(flatStackedFeatures, len(self.filterSizes)*self.numFilters, self.fcSize, True)
        dropFeatures = tf.nn.dropout(fc1Features, 1.0)
        fc2Features = self.fc_layer(dropFeatures, self.fcSize, self.numClasses, False)

        predictions = tf.argmax(fc2Features, 1)
        correct_predictions = tf.equal(predictions, tf.argmax(self.inputDict['classes'], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        return accuracy, self.inputDict

    def validation(self, accuracy, inputDict):
        self.accuracy = accuracy
        self.inputDict = inputDict
        saver = tf.train.Saver()
        with open('lb') as f:
            self.lb = pickle.load(f)
        if os.path.isfile('model/save.npy'):
            self.current_epoch, self.current_step = np.load(
                "model/save.npy")
        else:
            print "No Checkpoints Available, Train first!"
            sys.exit(0)
        ckpt_file = "./model/model.ckpt-" + str(self.current_step)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_file)
            batchIter = self.get_val_batch()
            for batch in xrange(0, len(self.XTest), self.batchSize):
                textBatch, actualClass = batchIter.next()
                feedDict = {}
                feedDict[self.inputDict['text']] = textBatch
                feedDict[self.inputDict['classes']] = actualClass
                acc = sess.run(self.accuracy, feed_dict=feedDict)
                print "Validation Batch:", batch + 1, "\tAccuracy: ", acc
