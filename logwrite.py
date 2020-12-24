import datetime
from tensorflow.keras.callbacks import Callback


class Logging_progress(Callback):

    def __init__(self, mod, turn):
        super().__init__()
        self.mod = mod
        self.path = '/home/tensorflow/word2vec/logs/' + mod
        self.turn = turn

    def on_train_begin(self, logs=None):
        with open(self.path+f'/input_{self.turn}.log', 'a') as log_file:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            log_file.write('['+now+']\t'+f'Start Training target_{self.turn}.npy file'+'\n')

    def on_train_end(self, logs=None):
        with open(self.path+f'/input_{self.turn}.log', 'a') as log_file:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            log_file.write('['+now+']\t'+f'End Training target_{self.turn}.npy file'+'\n')

    def on_epoch_begin(self, epoch, logs=None):
        with open(self.path+f'/input_{self.turn}.log', 'a') as log_file:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            log_file.write('['+now+']\t'+f'Start Epoch: {epoch+1}'+'\n')

    def on_epoch_end(self, epoch, logs=None):
        with open(self.path+f'/input_{self.turn}.log', 'a') as log_file:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            loss = "{:7.2f}".format(logs["loss"])
            acc = "{:7.2f}".format(logs["accuracy"])
            log_file.write('['+now+']\t'+f'End Epoch: {epoch+1}, loss: {loss}, acc: {acc}'+'\n')
        with open('/home/tensorflow/word2vec/checkpoints/' + self.mod + '/checkpoint', 'r+') as f:
            old = f.read()
            a = old[old.find("cp-"):old.find('.ckpt')]
            b = "cp-" + str(epoch+1).zfill(4)
            new = old.replace(a, b)
            f.seek(0)
            f.write(new)
