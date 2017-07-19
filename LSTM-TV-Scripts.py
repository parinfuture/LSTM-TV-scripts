import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]

import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # No duplicate words to be allowed as part of the mappings
    vocab_to_int = {word: i for i, word in enumerate(set(text), 0)}
    int_to_vocab = {i: word for i, word in enumerate(set(text), 0)}
    return vocab_to_int, int_to_vocab



"""
Test passed?
"""
tests.test_create_lookup_tables(create_lookup_tables)

#Tokenizing punctuation
short_words = {}
vocab_to_int, int_to_vocab = create_lookup_tables(text)
for word in vocab_to_int:
    if len(word)<3:
        if word in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":
            continue
        short_words[word] = ""
print(short_words)

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # Is there a better way to do this, seems I have duplicated a lot of code? 
    msg = '||{}||'
    punctuation_tuples = [
        ('.', msg.format('period')), (',', msg.format('comma')), ('"', msg.format('quotation_mark')),
        (';', msg.format('semicolon')), ('!', msg.format('exclamation_mark')), ('?', msg.format('question_mark')),
        ('(', msg.format('left_parentheses')), (')', msg.format('right_parentheses')), ('--', msg.format('dash')),
        ('\n', msg.format('return'))
    ]
    return dict(punctuation_tuples)

"""
Test passed?
"""
tests.test_tokenize(token_lookup)

#Preprocessing all data and saving it
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

#Checking the version of Tensorflow and Access to GPU

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


#Create get_inputs() to create TF placeholders
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    Input = tf.placeholder(tf.int32, [None, None], name='input')
    Targets = tf.placeholder(tf.int32, [None, None])
    LearningRate = tf.placeholder(tf.float32)
    return Input, Targets, LearningRate

#Build RNN Cell and initiate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    layers_rnn = 2
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * layers_rnn)
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    return cell, initial_state

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # Need to provide dtype if initial state is not provided
    Outputs, State = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    FinalState = tf.identity(State, name='final_state')
    return Outputs, FinalState

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, FinalState = build_rnn(cell, embed)
    Logits = tf.contrib.layers.fully_connected(outputs, num_outputs=vocab_size, activation_fn=None)
    return Logits, FinalState

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    index = 0
    first_element = int_text[0]
    number_of_batches = int((len(int_text) // batch_size) / seq_length)
    const = int(number_of_batches * batch_size * seq_length)
    batches = np.zeros([number_of_batches, 2, batch_size, seq_length], dtype=np.int32)
    for i in range(number_of_batches):
        # These are the batches representing the each single batch
        # and then the input and the target representation of that as well
        for j in range(batch_size):
            index = (i * seq_length) + (j * number_of_batches * seq_length) # move to the right by the number of batches
            inputs = np.array(int_text[index:index + seq_length])
            target = np.array(int_text[index + 1:index + seq_length + 1])
            if i == number_of_batches - 1 and j == batch_size - 1:
                target[seq_length - 1] = first_element
            batches[i][0][j] = inputs
            batches[i][1][j] = target
    return batches

#Hyperparameters

# Number of Epochs
num_epochs = 80
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 200
# Embedding Dimension Size
embed_dim = 200
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 100

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')

# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))

#Checkpoint

import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()


#Implmenet generate functions
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name('input:0')
    InitialStateTensor = loaded_graph.get_tensor_by_name('initial_state:0')
    FinalStateTensor = loaded_graph.get_tensor_by_name('final_state:0')
    ProbsTensor = loaded_graph.get_tensor_by_name('probs:0')
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor

#Choose word

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    x = np.random.choice(len(int_to_vocab), 1, p=probabilities)[0]
    # TODO: Implement Function
    return int_to_vocab[x]

#Generate TV script

gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'


loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)