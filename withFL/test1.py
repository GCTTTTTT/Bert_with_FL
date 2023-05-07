import time
import tensorflow as tf
import numpy as np
from collections import defaultdict
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

from bert import modeling
from bert import optimization
from bert import tokenization

from federated_utils import read_federated_train_data, read_vocab_and_labels, create_masked_lm_predictions, serialize_instances, preprocess_data, write_serialized_data

# 联邦学习相关配置
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 2
LEARNING_RATE = 1e-5

# BERT 预训练相关配置
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 128
MASKED_LM_PROB = 0.15
RANDOM_SEED = 42
TRAIN_STEPS = 1000
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 1000
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
USE_TPU = False

input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_ids')
input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='input_mask')
segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH], name='segment_ids')
masked_lm_positions = tf.placeholder(tf.int32, [None, None], name='masked_lm_positions')
masked_lm_ids = tf.placeholder(tf.int32, [None, None], name='masked_lm_ids')
masked_lm_weights = tf.placeholder(tf.float32, [None, None], name='masked_lm_weights')
next_sentence_labels = tf.placeholder(tf.int32, [None], name='next_sentence_labels')

bert_config = modeling.BertConfig.from_json_file('bert_config.json')
model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

masked_lm_loss = -tf.reduce_mean(masked_lm_example_loss)
next_sentence_loss = tf.reduce_mean(next_sentence_example_loss)

total_loss = masked_lm_loss + next_sentence_loss
train_op = optimization.create_optimizer(
            total_loss,
            LEARNING_RATE,
            TRAIN_STEPS,
            NUM_TRAIN_STEPS_PER_DECAY,
            ITERATIONS_PER_LOOP,
            use_tpu=USE_TPU)

# 读取词典和标签
vocab, labels = read_vocab_and_labels(VOCAB_FILE, LABELS_FILE)

# 随机种子
np.random.seed(RANDOM_SEED)

with tf.Session() as sess:
    # 为 BERT 模型设置 RESTORE 对应的变量
    restore_vars = tf.global_variables()
    tvars = tf.trainable_variables()
    assignment_map, uninitialized_vars = modeling.get_assignment_map_from_checkpoint(tvars, INIT_CHECKPOINT)
    tf.train.init_from_checkpoint(INIT_CHECKPOINT, assignment_map)

    # 初始化其他变量
    sess.run(tf.variables_initializer([i for i in restore_vars if i not in uninitialized_vars]))
    sess.run(tf.local_variables_initializer())

    # 分批次从联邦学习客户端读取数据并训练模型
    for r in range(NUM_ROUNDS):
        np.random.shuffle(client_ids)
        for i in range(0, len(client_ids), CLIENTS_PER_ROUND):
            client_batch = client_ids[i:i + CLIENTS_PER_ROUND]
            print(f'Running round {r + 1}, clients {i}-{i + CLIENTS_PER_ROUND}')

            train_data = read_federated_train_data(client_batch, DATA_DIR)
            preproc_train_data = preprocess_data(train_data)
            serialized_data = serialize_instances(preproc_train_data, MASKED_LM_PROB, vocab, labels, MAX_SEQ_LENGTH)
            write_serialized_data(serialized_data, i)

            input_fn = input_fn_builder(
                input_file=f'{i}',
                seq_length=MAX_SEQ_LENGTH,
                is_training=True,
                batch_size=BATCH_SIZE)

            sess.run(tf.tables_initializer())
            sess.run(tf.global_variables_initializer())

            for j in range(TRAIN_STEPS):
                loss, _ = sess.run([total_loss, train_op], feed_dict={input_ids: input_ids_batch,
                                                                      input_mask: input_mask_batch,
                                                                      segment_ids: segment_ids_batch,
                                                                      masked_lm_positions: masked_lm_positions_batch,
                                                                      masked_lm_ids: masked_lm_ids_batch,
                                                                      masked_lm_weights: masked_lm_weights_batch,
                                                                      next_sentence_labels: next_sentence_labels_batch})
                if j % DISPLAY_STEPS == 0:
                    print(f'step: {j}, loss: {loss}')

            print(f'Round {r + 1} finished')


