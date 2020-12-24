import tensorflow as tf
import tqdm
import re
import string
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import json
import math


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


def get_vocab_size(path_to_file):
    vocab, vocab_size = {}, 1  # start indexing from 1
    vocab['<pad>'] = 0  # add a padding token
    with open(path_to_file) as f:
        lines = list(sentence.split() for sentence in f.read().splitlines())
        for line in lines:
            for token in line:
                if token not in vocab:
                    vocab[token] = vocab_size
                    vocab_size += 1
        vocab_size = len(vocab)
    return vocab_size


def generate_training_data(window_size, num_ns, seed, batch_size, input_file, mod):
    text_ds = tf.data.TextLineDataset(input_file).filter(lambda x: tf.cast(tf.strings.length(x) >= 2, bool))
    vocab_size = get_vocab_size(input_file)

    sequence_length = len(
        max(list(x.numpy().decode('utf-8') for x in list(text_ds)), key=lambda n: len(n.split())).split())

    print(f"sequence_length: {sequence_length}, vocab_size: {vocab_size}")
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)
    vectorize_layer.adapt(text_ds.batch(batch_size))
    vocab = {k: v for v, k in enumerate(vectorize_layer.get_vocabulary())}
    with open('resource/model_input/'+mod+'/vocab.json', 'w', encoding='UTF-8') as f:
        json.dump(vocab, f)
    text_vector_ds = text_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vector_ds.as_numpy_iterator())

    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in tqdm.tqdm(sequences):

        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0] * num_ns, dtype="int64")

            a = context.numpy()

            targets.append(target_word)
            contexts.append(context.numpy())
            labels.append(label.numpy())

    count = math.floor(len(targets)/14)
    # for i in range(0,14):
    #     np.save(f'resource/model_input/'+mod+f'/targets_{i + 1}', np.array(targets[i * count:(i + 1) * count]))
    #     np.save(f'resource/model_input/'+mod+f'/contexts_{i + 1}', np.array(contexts[i * count:(i + 1) * count]))
    #     np.save(f'resource/model_input/'+mod+f'/labels_{i + 1}', np.array(labels[i * count:(i + 1) * count]))
    #
    # np.save('resource/model_input/'+mod+f'/targets_15', np.array(targets[14 * count:]))
    # np.save('resource/model_input/'+mod+f'/contexts_15', np.array(targets[14 * count:]))
    # np.save('resource/model_input/'+mod+f'/labels_15', np.array(targets[14 * count:]))

    return targets, contexts, labels, vocab
