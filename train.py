import tensorflow as tf
import json
import numpy as np
from model import Word2Vec
import os, io, sys, math
from logwrite import Logging_progress
import gen_data as gd

settings = {
    'SEED': 42,
    'window_size': 5,
    'num_ns': 4,
    'BATCH_SIZE': 512,
    'BUFFER_SIZE': 10000,
    'embedding_dim': 200,
    'epoch': 7
}

mod = 'all'

ckpt_data = '/cp-{epoch}.ckpt.data-00000-of-00001'
ckpt_idx = '/cp-{epoch}.ckpt.index'

filenames = {
    'naver_dic': '/home/tensorflow/word2vec/resource/split_txt/naver_dic.txt',
    'data_set': '/home/tensorflow/word2vec/resource/split_txt/word.txt',
    'all': '/home/tensorflow/word2vec/resource/split_txt/input.txt'
}

if __name__ == '__main__':
    # targets, contexts, labels, vocab = gd.generate_training_data(
    #     window_size=settings['window_size'],
    #     num_ns=settings['num_ns'],
    #     seed=settings['SEED'],
    #     batch_size=settings['BATCH_SIZE'],
    #     input_file=filenames[mod],
    #     mod=mod
    # )

    checkpoint_path = '/home/tensorflow/word2vec/checkpoints/' + mod + '/cp-{epoch:04d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    with open('/home/tensorflow/word2vec/resource/model_input/' + mod + '/vocab.json') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    word2vec = Word2Vec(vocab_size, settings['embedding_dim'], settings['num_ns'])
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy']
                     )

    try:
        turn = int(math.ceil(int(latest[len(checkpoint_dir) + 4:-5]) / settings['epoch']))+1
        if turn == 16: sys.exit()
        for i in range((turn - 2) * settings['epoch'], (turn - 1) * settings['epoch']-1):
            data = checkpoint_dir + ckpt_data.format(epoch=str(i).zfill(4))
            idx = checkpoint_dir + ckpt_idx.format(epoch=str(i).zfill(4))
            if os.path.isfile(data) and os.path.isfile(idx):
                os.remove(data)
                os.remove(idx)
        init_epoch = int(latest[len(checkpoint_dir) + 4:-5])
        end_epoch = turn * settings['epoch']
        word2vec.load_weights(latest)
        print(f'Starting {latest}')
    except:
        turn = 1
        init_epoch = 1
        end_epoch = settings['epoch']

    targets = np.load('/home/tensorflow/word2vec/resource/model_input/' + mod + f'/targets_{turn}.npy')
    targets = list(np.reshape(targets, (-1, 1)))
    contexts = list(np.load('/home/tensorflow/word2vec/resource/model_input/' + mod + f'/contexts_{turn}.npy'))
    labels = list(np.load('/home/tensorflow/word2vec/resource/model_input/' + mod + f'/labels_{turn}.npy'))

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(settings['BUFFER_SIZE']).batch(settings['BATCH_SIZE'], drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )
    logging_callback = Logging_progress(mod, turn)

    word2vec.fit(
        dataset,
        epochs=end_epoch,
        callbacks=[checkpoint_callback, logging_callback],
        initial_epoch=init_epoch
    )

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    out_v = io.open('/home/tensorflow/word2vec/result/' + mod + f'/vectors_{turn}.tsv', 'w', encoding='utf-8')
    out_m = io.open('/home/tensorflow/word2vec/result/' + mod + f'/metadata_{turn}.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0: continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
