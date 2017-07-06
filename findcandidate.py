#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import json
from tqdm import tqdm
import random
import operator
import math
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/opt/cnn-text-classification-tf/runs/1498819839/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("output", "/opt/cnn-text-classification-tf/predict_res.1w.txt", "output file")
tf.flags.DEFINE_string("input", "/data01/cnn-predseq/positive.txt", "input file")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

#load data
print "\nLoading Data..\n"
input_file = open(FLAGS.input,'r')
input_seqs = []
seen = {}
for line in input_file:
    if not line:
        break
    line = line.strip()
    input_seq = line.split(' ')
    for x in input_seq:
        seen[x] = True
    input_seqs.append(input_seq)
input_file.close()

output_file = open(FLAGS.output,'w')
all_vids = [x for x in seen]


# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        #predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        predictions = graph.get_operation_by_name("output/scores").outputs[0]
        input_seqs = random.sample(input_seqs,10000)
        for seq in tqdm(input_seqs):    
            eval_set = []
            candidates = all_vids #random.sample(all_vids,10000)
            for next_vid in candidates:
                pred_seq = seq + [next_vid]
                pred_seq = " ".join(pred_seq)
                eval_set.append(pred_seq)
            x_eval = np.array(list(vocab_processor.transform(eval_set)))                
            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)
            all_predictions = []                        
            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                normalized_preds = []
                for pred in batch_predictions:
                    max_val = max(pred[0],pred[1])
                    prob0 = math.exp(pred[0] - max_val)
                    prob1 = math.exp(pred[1] - max_val)
                    prob1 = prob1/(prob0+prob1)
                    normalized_preds = normalized_preds + [prob1]
                #batch_predictions = [x[1] for x in batch_predictions]
                all_predictions = np.concatenate([all_predictions, normalized_preds])
            pred_res = zip(candidates,all_predictions)
            pred_res.sort(key=operator.itemgetter(1),reverse=True)
            pred_res = pred_res[:100]
            #save result
            key = ','.join([str(x) for x in seq])
            top_candidates = ';'.join(['{},{}'.format(x[0],x[1]) for x in pred_res])
            line = '{}\t{}\n'.format(key,top_candidates)
            output_file.write(line)
        output_file.close()

print 'Done!'
print 'Check file ',FLAGS.output
