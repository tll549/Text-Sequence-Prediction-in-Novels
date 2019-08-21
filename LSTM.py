#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#LSTM" data-toc-modified-id="LSTM-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>LSTM</a></span></li><li><span><a href="#Code" data-toc-modified-id="Code-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Code</a></span></li><li><span><a href="#Testing-parameters" data-toc-modified-id="Testing-parameters-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Testing parameters</a></span><ul class="toc-item"><li><span><a href="#Two-embedding-methods" data-toc-modified-id="Two-embedding-methods-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Two embedding methods</a></span></li><li><span><a href="#n_hidden" data-toc-modified-id="n_hidden-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>n_hidden</a></span></li><li><span><a href="#Learning-rate" data-toc-modified-id="Learning-rate-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Learning rate</a></span></li></ul></li><li><span><a href="#Final-model" data-toc-modified-id="Final-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Final model</a></span><ul class="toc-item"><li><span><a href="#Most-frequently-right-and-wrong-words" data-toc-modified-id="Most-frequently-right-and-wrong-words-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Most frequently right and wrong words</a></span></li><li><span><a href="#Prediction-on-test-set" data-toc-modified-id="Prediction-on-test-set-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Prediction on test set</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>References</a></span></li></ul></div>

# # LSTM
# 
# **Name: Yi Hung Liu**
#    
# Submitted files
# 
# - .pdf
# - .ipynb
# - .py
# - .txt
# 
# # Code

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import string
import time
import matplotlib.pyplot as plt


# In[2]:


def read_data():
    train = []
    val = []
    test = []
    with open("bobsue-data/bobsue.seq2seq.train.tsv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            train.append(row[0])
    with open("bobsue-data/bobsue.seq2seq.dev.tsv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            val.append(row[0])
    with open("bobsue-data/bobsue.seq2seq.test.tsv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            test.append(row[0])
            
    print(len(train))
    print(len(val))
    print(len(test))
    
    return(train, val, test)

def preprocess_data(data):
    """lower"""
    data = [d.lower() for d in data]
    return(data)

train, val, test = read_data()
train = preprocess_data(train)
val = preprocess_data(val)
test = preprocess_data(test)


# - `preprocess_data()` makes every letters lower.

# In[3]:


def gen_dict():
    # make dict
    # including all punctuations, numbers, symbols, <s> and </s>
    words = []
    for i in range(len(train)):
        for word in train[i].split():
            words.append(word)
    for i in range(len(val)):
        for word in val[i].split():
            words.append(word)
    for i in range(len(test)):
        for word in test[i].split():
            words.append(word)
    print("words count:", len(words))
    print("unique words:", len(set(words)))

    idx2word = {idx:word for idx, word in enumerate(set(words))}
    word2idx = {word:idx for idx, word in enumerate(set(words))}
    return(idx2word, word2idx, words)
idx2word, word2idx, words = gen_dict()

def rand_embed():
    torch.manual_seed(666)
    # idx2feat = {idx:torch.rand(1, 200) for idx, word in enumerate(set(words))}
    idx2feat = {idx:torch.rand(1, 50) for idx, word in enumerate(set(words))}
    word2feat = {word:idx2feat[idx] for idx, word in enumerate(set(words))}
    return(idx2feat, word2feat)
idx2feat, word2feat = rand_embed()

def prepare_data(data):
    processed_data = []
    for d in data:
        d = d.split()
        d = [x for x in d if x != '<s>'] # remove every <s>
        processed_data.append(d)
    return(processed_data)
train_input = prepare_data(train)
val_input = prepare_data(val)
test_input = prepare_data(test)


# - `gen_dict()` generate a dictionary mapping between indices and words
# - `rand_embed()` generate random embedding tensors in 50 dimension space (I choose 50 to increase the speed of converges)
# - `prepare_data()` split every line of data into list and remove <s>

# In[175]:


def pre_competed_embedding():
    word2pce_temp = {} # pre-computed embedding
    # with open("glove.6B.200d.txt", 'r') as f:
    with open("glove.6B.50d.txt", 'r') as f:
        for l in f:
            line = l.split()
            word2pce_temp[line[0]] = torch.Tensor(np.array(line[1:]).astype(np.float)) # shape = 50
            word2pce_temp[line[0]] = word2pce_temp[line[0]].unsqueeze(0) # shape = 1*50
    print("total num of keys:", len(word2pce_temp))

    word2pce = {}
    not_included = []
    for key in word2idx:
        if key in word2pce_temp:
            word2pce[key] = word2pce_temp[key]
        else:
            word2pce[key] = word2feat[key]
            not_included.append(key)
    print("num of selected keys:", len(word2pce))
    print("not included, use rand instead:", not_included)
    return(word2pce)
word2pce = pre_competed_embedding()

def word2tens(w, embed_type):
    if embed_type == "rand":
        return(word2feat[w][:])
    else: # pce
        return(word2pce[w][:])


# In[176]:


print(word2pce['the'].shape)
print(word2pce["'t"].shape)


# - `pre_competed_embedding()` load the pre-computed 50 dimension GloVe embedding from [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) and makes it as a dictionary.
# - `word2tens()` conver a word (string) to its corresponding embedding tensor, depending on the `embed_type`

# In[142]:


def find_pred_word(y_pred, embed_type = "rand"):
    """map 1*200 tensor to a word"""
    if embed_type == "rand":
        all_words_distance = [torch.dist(y_pred, idx2feat[i]) for i in range(len(idx2feat))]
    else: # pce
        all_words_distance = [torch.dist(y_pred, list(word2pce.values())[i]) for i in range(len(word2pce))]
    return(idx2word[np.argmin(np.array(all_words_distance))])
    
def separate_sentence(d):
    first = []
    second = []
    doing = 1
    for i in range(len(d)):
        if doing == 1: first.append(d[i])
        else: second.append(d[i])
        if d[i] == '</s>': doing = 2
    return(first, second)

def print_list(input_list):
    out = ""
    for s in input_list:
        out += s + " "
    return(out)

def calc_acc(pred, target):
    acc = 0
    for i in range(len(pred)):
        if pred[i] == target[i]:
            acc += 1
    return(acc/len(target)) # if pred shorter than target (end by </s> earlier), the less part is wrong


# Helper functions:
# 
# - `find_pred_word()` convert a tensor to a word by find the minimum distance between all words.
# - `separate_sentence()` separate a standard line (2 sentences) as input_sentence and target_sentence

# In[252]:


# class myLSTM(nn.Module):
#     def __init__(self, n_in, n_hid, n_out):
#         super(myLSTM, self).__init__()
#         self.n_hid = n_hid
        
#         self.w_if = torch.randn(n_hid, n_in, requires_grad=True)
#         self.w_hf = torch.randn(n_hid, n_hid, requires_grad=True)
#         self.b_f = torch.randn(n_hid, requires_grad=True)
#         self.w_ii = torch.randn(n_hid, n_in, requires_grad=True)
#         self.w_hi = torch.randn(n_hid, n_hid, requires_grad=True)
#         self.b_i = torch.randn(n_hid, requires_grad=True)
#         self.w_io = torch.randn(n_hid, n_in, requires_grad=True)
#         self.w_ho = torch.randn(n_hid, n_hid, requires_grad=True)
#         self.b_o = torch.randn(n_hid, requires_grad=True)
#         self.w_ic = torch.randn(n_hid, n_in, requires_grad=True)
#         self.w_hc = torch.randn(n_hid, n_hid, requires_grad=True)
#         self.b_c = torch.randn(n_hid, requires_grad=True)
        
#         self.fc = nn.Linear(n_hid, n_out) #???
       
#     def lstmcell(self, x, h_t_1, c_t_1):
#         '''c_t_1 is c_{t-1}, h_t_1 is h_{t-1}'''
#         f_t = self.sigmoid(torch.matmul(self.w_if, x) + torch.matmul(self.w_hf, h_t_1) + self.b_f)
#         i_t = self.sigmoid(torch.matmul(self.w_ii, x) + torch.matmul(self.w_hi, h_t_1) + self.b_i)
#         o_t = self.sigmoid(torch.matmul(self.w_io, x) + torch.matmul(self.w_ho, h_t_1) + self.b_o)
#         c_t_tilda = torch.tanh(torch.matmul(self.w_ic, x) + torch.matmul(self.w_hc, h_t_1) + self.b_c)
#         c_t = f_t * c_t_1 + i_t * c_t_tilda # * is elementwise multiplication
#         h_t = o_t * torch.tanh(c_t)
#         return(c_t, h_t)
    
#     def init_hidden(self):
#         return torch.zeros(self.n_hid)
    
#     def forward(self, x, hidden=None, ctx=None):
#         if hidden is None:
#             hidden = self.init_hidden()
#             ctx = self.init_hidden()
            
#         hidden, ctx = self.lstmcell(x, hidden, ctx)
#         output = self.fc(hidden)
#         return output, (hidden, ctx)
    
#     def sigmoid(self, x):
#         return 1/(1 + torch.exp(-x))


# In[292]:


# def run_train(data, sample_verbose = 0, embed_type = "rand"):
#     losses = []
#     acc = []
#     for d in data: # every sentence
#         input_sentence, target_sentence = separate_sentence(d)
        
#         # learn and predict every words
#         pred_sentence = [d[0]]
#         hidden = None
#         ctx = None
#         for i in range(len(d)-1): # every word
#             to_predict_feat = word2tens(d[i], embed_type)
#             target_feat = word2tens(d[i+1], embed_type)
                
#             pred_feat, (hidden, ctx) = model(to_predict_feat.squeeze(0), hidden, ctx)
#             loss = mse(pred_feat, target_feat)
#             losses.append(loss.item())
            
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True) #???
#             optimizer.step()
#             # print('    input {:10} | predict {:10} | actual {:10} | loss {:.2f}'.format(d[i], find_pred_word(pred_feat), d[i+1], loss.item()))

#             pred_word = find_pred_word(pred_feat, embed_type = embed_type)
#             pred_sentence.append(pred_word)
#             if pred_word == '</s>' and i > len(input_sentence):
#                 break
    
#         # only evaluate the second part of the sentence for verbose, acc, count correctness
#         pred_target_sentence = pred_sentence[len(input_sentence):]
#         if sample_verbose > 0:
#             print('    train predicted:', print_list(pred_target_sentence), 
#                   '\n             actual:', print_list(target_sentence))
#             sample_verbose -=1
#         for i in range(len(target_sentence)):
#             if i < len(pred_target_sentence) and target_sentence[i] == pred_target_sentence[i]: # correct
#                 acc.append(1)
#                 if target_sentence[i] in most_right:
#                     most_right[target_sentence[i]] += 1
#                 else:
#                     most_right[target_sentence[i]] = 1
#             else:
#                 acc.append(0)
#                 if target_sentence[i] in most_wrong:
#                     most_wrong[target_sentence[i]] += 1
#                 else:
#                     most_wrong[target_sentence[i]] = 1
#     return(np.mean(losses), np.mean(acc))    

# def run_test(data, sample_verbose = 2, embed_type = "rand", return_all_pred = False):
#     losses = []
#     acc = []
#     all_pred = []
#     for d in data: # every sentence
#         input_sentence, target_sentence = separate_sentence(d)
#         pred_sentence = [d[0]]
#         hidden = None
#         ctx = None
        
#         # first part input sentence, try to predict but don't learn it!
#         for i in range(len(input_sentence)-1):
#             to_predict_feat = word2tens(d[i], embed_type)
#             target_feat = word2tens(d[i+1], embed_type)

#             pred_feat, (hidden, ctx) = model(to_predict_feat.squeeze(0), hidden, ctx)
            
#         # only evaluate the second part of the sentence for verbose, acc, count correctness
#         i = 0
#         target_feat = word2tens('</s>', embed_type) # start predict from this
#         while (True):
#             pred_feat, (hidden, ctx) = model(target_feat.squeeze(0), hidden, ctx)
            
#             # if it keep predicting when the actual target is finish, the redundant part are counted as 0.5 loss per word, but it won't be back prop 
#             if i < len(target_sentence): # still can compare to target
#                 target_feat = word2tens(d[i+1], embed_type)
#                 loss = mse(pred_feat, target_feat).item()
#             else:
#                 loss = 0.5
#                 target_feat = pred_feat # use from next prediction
#             losses.append(loss)
            
#             pred_word = find_pred_word(pred_feat, embed_type = embed_type)
#             pred_sentence.append(pred_word)
            
#             # print('i', i, 'input', find_pred_word(to_predict_feat, embed_type), 'pred', pred_word, 'target', find_pred_word(target_feat, embed_type = embed_type), 'loss', loss)
#             # to_predict_feat = pred_feat # for next iteration
#             i += 1
#             # if pred_word == '</s>' or i > 15:
#             if pred_word == '</s>' or len(pred_sentence) == len(target_sentence):
#                 break
        
#         # calc acc and most_right and most_wrong
#         if sample_verbose > 0:
#             print('    test  predicted:', print_list(pred_sentence), 
#                   '\n             actual:', print_list(target_sentence))
#             sample_verbose -=1
#         for i in range(len(target_sentence)):
#             if i < len(pred_sentence) and target_sentence[i] == pred_sentence[i]: # correct
#                 acc.append(1)
#             else:
#                 acc.append(0)
#         all_pred.append(pred_sentence)
        
#     if return_all_pred == False:
#         return(np.mean(losses), np.mean(acc)) 
#     if return_all_pred == True:
#         return(np.mean(losses), np.mean(acc), all_pred)

# def run_everything(plot = True, train_verbose = 0, test_verbose = 1, embed_type = "rand"):
#     global model, mse, optimizer
#     model = myLSTM(hparams['n_input'], hparams['n_hidden'], hparams['n_output'])
#     mse = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr = hparams['learning_rate'])

#     start = time.time()
#     epoch_train_losses = []
#     epoch_test_losses = []
#     epoch_train_acc = []
#     epoch_test_acc = []
#     global most_right, most_wrong
#     most_right = {}
#     most_wrong = {}
#     print('start training...')
#     for e in range(hparams['epochs']):
#         train_losses, train_acc = run_train(train_input[0:hparams['train_size']], sample_verbose = train_verbose, embed_type = embed_type)
#         epoch_train_losses.append(train_losses)
#         epoch_train_acc.append(train_acc)

#         test_losses, test_acc = run_test(val_input[0:hparams['test_size']], sample_verbose = test_verbose, embed_type = embed_type)
#         epoch_test_losses.append(test_losses)
#         epoch_test_acc.append(test_acc)

#         if e % hparams['logint'] == 0:
#             elapsed = (time.time() - start)
#             print('Epoch {:3} | Train Loss: {:3.3f} | Test Loss {:6.3f} | Train Acc: {:4.2f} | Test Acc: {:4.2f} | Running Time: {:7.2f}'.format(
#                 e, epoch_train_losses[-1], epoch_test_losses[-1], train_acc, test_acc, elapsed))
    
#     if plot:
#         plt.plot(range(len(epoch_train_losses)), epoch_train_losses)
#         plt.plot(range(len(epoch_test_losses)), epoch_test_losses)
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(('train', 'val'))
#         plt.show()
        
#         plt.plot(range(len(epoch_train_acc)), epoch_train_acc)
#         plt.plot(range(len(epoch_test_acc)), epoch_test_acc)
#         plt.ylabel('Acc')
#         plt.xlabel('Epoch')
#         plt.legend(('train', 'val'))
#         plt.show()


# In[143]:


class MyLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(n_in, n_hidden)
        self.fc = nn.Linear(n_hidden, n_out)
        self.n_hidden = n_hidden
        
    def forward(self, x):
        hiddens, (last_h, last_ctx) = self.lstm(x)
        return self.fc(last_h).squeeze(0)
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.n_hidden),
                torch.zeros(1, 1, self.n_hidden))


# In[192]:


def run():
    if hparams['rand_seed'] != 0:
        torch.manual_seed(hparams['rand_seed'])
        
    model = MyLSTM(hparams['n_input'], hparams['n_hidden'], hparams['n_output'])
    mse = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=hparams['learning_rate'])     

    start = time.time()
    epoch_train_losses = []
    epoch_train_acc = []
    epoch_test_losses = []
    epoch_test_acc = []
    print('start training...')
    for e in range(hparams['epochs']):
        # train
        train_losses = []
        train_acc = []    
        for d_idx in range(hparams['train_size']): # every sentence
            d = train_input[d_idx]
            if d[-1] != '</s>': # don't know why some data incomplete
                continue
            model.hidden = model.init_hidden()
            first, second = separate_sentence(d)
            pred = [d[0]]
            for i in range(len(d)-1): # every word
                input_feat = word2tens(d[i], hparams['embedding'])[:]            
                target_feat = word2tens(d[i+1], hparams['embedding'])[:]
                pred_feat = model(input_feat.unsqueeze_(0))

                model.zero_grad()
                loss = mse(target_feat, pred_feat)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                pred.append(find_pred_word(pred_feat))

            pred_first = pred[0:len(first)]
            pred_second = pred[len(first):]
            current_acc = calc_acc(pred_second, second)
            train_acc.append(current_acc)
            if hparams['print_pred'] != 0:
                if d_idx % round(hparams['train_size'] / hparams['print_pred']) == 0:
                    print('    training:', d_idx, 'acc:', round(current_acc, 3),
                          '\n        predicted:', print_list(pred_second), 
                          '\n           actual:', print_list(second))
        epoch_train_losses.append(np.mean(train_losses))
        epoch_train_acc.append(np.mean(train_acc))

        # test
        test_losses = []
        test_acc = []    
        for d_idx in range(hparams['test_size']): # every sentence
            d = test_input[d_idx]
            if d[-1] != '</s>': # don't know why some data incomplete
                continue
            model.hidden = model.init_hidden()
            first, second = separate_sentence(d)
            pred = [d[0]]
            for i in range(len(d)-1): # every word
                input_feat = word2tens(d[i], hparams['embedding'])[:]            
                target_feat = word2tens(d[i+1], hparams['embedding'])[:]
                pred_feat = model(input_feat.unsqueeze_(0))

                loss = mse(target_feat, pred_feat)

                # only calculate loss for second part
                if i >= len(first):
                    test_losses.append(loss.item())
                pred_word = find_pred_word(pred_feat)
                pred.append(pred_word)
                if i > len(first) and pred_word == '</s>':
                    break

            pred_first = pred[0:len(first)]
            pred_second = pred[len(first):]
            current_acc = calc_acc(pred_second, second)
            test_acc.append(current_acc)
            if hparams['print_pred'] != 0:
                if d_idx % round(hparams['train_size'] / hparams['print_pred']) == 0:
                    print('    testng:', d_idx, 'acc:', round(current_acc, 3),
                          '\n        predicted:', print_list(pred_second), 
                          '\n           actual:', print_list(second))
        epoch_test_losses.append(np.mean(test_losses))
        epoch_test_acc.append(np.mean(test_acc))

        if e % hparams['logint'] == 0:
            elapsed = (time.time() - start)
            print('Epoch {:3} | Train Loss: {:3.3f} Acc {:3.3f} | Test Loss: {:3.3f} Acc {:3.3f} | RunTime: {:7.2f}'.format(e, epoch_train_losses[-1], epoch_train_acc[-1], epoch_test_losses[-1], epoch_test_acc[-1], elapsed))

    if hparams['plot']:
        plt.plot(range(len(epoch_train_losses)), epoch_train_losses)
        plt.plot(range(len(epoch_test_losses)), epoch_test_losses)
        plt.ylabel('Training Loss')
        plt.xlabel('Epoch')
        plt.legend(('train', 'val'))
        plt.show()

        plt.plot(range(len(epoch_train_acc)), epoch_train_acc)
        plt.plot(range(len(epoch_test_acc)), epoch_test_acc)
        plt.ylabel('Training Acc')
        plt.xlabel('Epoch')
        plt.legend(('train', 'val'))
        plt.show()
    return(0)


# In[ ]:


# def run_train(data, sample_verbose = 0, embed_type = "rand"):
#     losses = []
#     acc = []
#     for d in data: # every sentence
#         input_sentence, target_sentence = separate_sentence(d)
        
#         # learn and predict every words
#         pred_sentence = [d[0]]
#         hidden = None
#         ctx = None
#         for i in range(len(d)-1): # every word
#             to_predict_feat = word2tens(d[i], embed_type)
#             target_feat = word2tens(d[i+1], embed_type)

#             pred_feat = model(to_predict_feat[:].unsqueeze_(0))
#             # pred_feat, (hidden, ctx) = model(to_predict_feat.squeeze(0), hidden, ctx)
#             loss = mse(pred_feat, target_feat)
#             losses.append(loss.item())
            
#             optimizer.zero_grad()
#             loss.backward(retain_graph=True) #???
#             optimizer.step()
#             # print('    input {:10} | predict {:10} | actual {:10} | loss {:.2f}'.format(d[i], find_pred_word(pred_feat), d[i+1], loss.item()))

#             pred_word = find_pred_word(pred_feat, embed_type = embed_type)
#             pred_sentence.append(pred_word)
#             if pred_word == '</s>' and i > len(input_sentence):
#                 break
    
#         # only evaluate the second part of the sentence for verbose, acc, count correctness
#         pred_target_sentence = pred_sentence[len(input_sentence):]
#         if sample_verbose > 0:
#             print('    train predicted:', print_list(pred_target_sentence), 
#                   '\n             actual:', print_list(target_sentence))
#             sample_verbose -=1
#         for i in range(len(target_sentence)):
#             if i < len(pred_target_sentence) and target_sentence[i] == pred_target_sentence[i]: # correct
#                 acc.append(1)
#                 if target_sentence[i] in most_right:
#                     most_right[target_sentence[i]] += 1
#                 else:
#                     most_right[target_sentence[i]] = 1
#             else:
#                 acc.append(0)
#                 if target_sentence[i] in most_wrong:
#                     most_wrong[target_sentence[i]] += 1
#                 else:
#                     most_wrong[target_sentence[i]] = 1
#     return(np.mean(losses), np.mean(acc))    

# def run_test(data, sample_verbose = 2, embed_type = "rand", return_all_pred = False):
#     losses = []
#     acc = []
#     all_pred = []
#     for d in data: # every sentence
#         input_sentence, target_sentence = separate_sentence(d)
#         pred_sentence = [d[0]]
#         hidden = None
#         ctx = None
        
#         # first part input sentence, try to predict but don't learn it!
#         for i in range(len(input_sentence)-1):
#             to_predict_feat = word2tens(d[i], embed_type)
#             target_feat = word2tens(d[i+1], embed_type)

#             # pred_feat, (hidden, ctx) = model(to_predict_feat.squeeze(0), hidden, ctx)
#             pred_feat = model(to_predict_feat[:].unsqueeze_(0))
            
#         # only evaluate the second part of the sentence for verbose, acc, count correctness
#         i = 0
#         target_feat = word2tens('</s>', embed_type) # start predict from this
#         while (True):
#             pred_feat = model(to_predict_feat[:].unsqueeze_(0))
#             # pred_feat, (hidden, ctx) = model(target_feat.squeeze(0), hidden, ctx)
            
#             # if it keep predicting when the actual target is finish, the redundant part are counted as 0.5 loss per word, but it won't be back prop 
#             if i < len(target_sentence): # still can compare to target
#                 target_feat = word2tens(d[i+1], embed_type)
#                 loss = mse(pred_feat, target_feat).item()
#             else:
#                 loss = 0.5
#                 target_feat = pred_feat # use from next prediction
#             losses.append(loss)
            
#             pred_word = find_pred_word(pred_feat, embed_type = embed_type)
#             pred_sentence.append(pred_word)
            
#             # print('i', i, 'input', find_pred_word(to_predict_feat, embed_type), 'pred', pred_word, 'target', find_pred_word(target_feat, embed_type = embed_type), 'loss', loss)
#             # to_predict_feat = pred_feat # for next iteration
#             i += 1
#             # if pred_word == '</s>' or i > 15:
#             if pred_word == '</s>' or len(pred_sentence) == len(target_sentence):
#                 break
        
#         # calc acc and most_right and most_wrong
#         if sample_verbose > 0:
#             print('    test  predicted:', print_list(pred_sentence), 
#                   '\n             actual:', print_list(target_sentence))
#             sample_verbose -=1
#         for i in range(len(target_sentence)):
#             if i < len(pred_sentence) and target_sentence[i] == pred_sentence[i]: # correct
#                 acc.append(1)
#             else:
#                 acc.append(0)
#         all_pred.append(pred_sentence)
        
#     if return_all_pred == False:
#         return(np.mean(losses), np.mean(acc)) 
#     if return_all_pred == True:
#         return(np.mean(losses), np.mean(acc), all_pred)

# def run_everything(plot = True, train_verbose = 0, test_verbose = 1, embed_type = "rand"):
#     global model, mse, optimizer
#     model = myLSTM(hparams['n_input'], hparams['n_hidden'], hparams['n_output'])
#     mse = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr = hparams['learning_rate'])

#     start = time.time()
#     epoch_train_losses = []
#     epoch_test_losses = []
#     epoch_train_acc = []
#     epoch_test_acc = []
#     global most_right, most_wrong
#     most_right = {}
#     most_wrong = {}
#     print('start training...')
#     for e in range(hparams['epochs']):
#         train_losses, train_acc = run_train(train_input[0:hparams['train_size']], sample_verbose = train_verbose, embed_type = embed_type)
#         epoch_train_losses.append(train_losses)
#         epoch_train_acc.append(train_acc)

#         test_losses, test_acc = run_test(val_input[0:hparams['test_size']], sample_verbose = test_verbose, embed_type = embed_type)
#         epoch_test_losses.append(test_losses)
#         epoch_test_acc.append(test_acc)

#         if e % hparams['logint'] == 0:
#             elapsed = (time.time() - start)
#             print('Epoch {:3} | Train Loss: {:3.3f} | Test Loss {:6.3f} | Train Acc: {:4.2f} | Test Acc: {:4.2f} | Running Time: {:7.2f}'.format(
#                 e, epoch_train_losses[-1], epoch_test_losses[-1], train_acc, test_acc, elapsed))
    
#     if plot:
#         plt.plot(range(len(epoch_train_losses)), epoch_train_losses)
#         plt.plot(range(len(epoch_test_losses)), epoch_test_losses)
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(('train', 'val'))
#         plt.show()
        
#         plt.plot(range(len(epoch_train_acc)), epoch_train_acc)
#         plt.plot(range(len(epoch_test_acc)), epoch_test_acc)
#         plt.ylabel('Acc')
#         plt.xlabel('Epoch')
#         plt.legend(('train', 'val'))
#         plt.show()

# hparams = {
#     'learning_rate': 0.1,
#     'epochs': 10,
#     'train_size': 5,
#     'test_size': 1,
#     'n_hidden': 100,
#     'n_input': 50,
#     'n_output': 50,
#     'logint': 1}  
# torch.manual_seed(666)
# run_everything(train_verbose = 1, test_verbose = 1, embed_type = "rand")


# My LSTM class function, following the same framework for the note of recitation. The notation is similar to [Understanding LSTM Networks -- colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) but separate the $W$ into two that act on $h$ and $x$.

# Main functions
# 
# - `run_train()` initial and train the model in the global envrionment, used in `run_everything()` for every epoch
#     - every word of every sentences are feeded one by one, the losses are calculated and back progagated
#     - returns loss and accuracy which calculated only on the second target part of a sentence
# - `run_test()` test the model just trained on validataion set or test set, used in `run_everything()` for every epoch
#     - it takes and try to remember every words in the first input part of a sentence, pretending it doesn't know the next incomeing word, so no updates on weights.
#     - then it predicts the next target sentence by starting from the </s> token (because I deleted all <s> so </s> can also be used as a start token for the next sentence
#     - it keeps predicting words until generate a stop token or the length exceeds 15
#     - the loss is calculated as the same way for the corresponding part of the predicted sentence and tartget sentence. However, the excess parts that longer than the target sentence are set as 0.5 loss per word. This is intended to see the loss change in final result. This arbitrary setting doesn't influence model learning because it won't be back propagated.
#     - after prediction, calculate and stroe the accuracy and occurance of words
# - `run_everything()` run the above two function in every epoch, generate print outs and plots
# 
# # Testing parameters
# 
# ## Two embedding methods

# In[184]:


hparams = {
    'learning_rate': 5,
    'epochs':        5,
    'train_size':    100, # max 6036
    'test_size':     20,  # max 750
    'n_hidden':      100,
    'n_input':       50,
    'n_output':      50,
    'logint':        1,   # frequency of printing result every epoch
    'print_pred':    2,   # frequency of printing sentence in one epoch, 0 don't print
    'plot':          True,
    'embedding':     'rand',
    'rand_seed':     555}
run()


# In[185]:


hparams['embedding'] = 'pce'
run()


# Above result prints the first example for every epoch. I use a small data set for training and testing parameters because of the slow speed and time limit.
# 
# As shown above, the random embedding converges quicker and has lower loss, which is surprising. But I will stick with it.
# 
# ## n_hidden

# In[193]:


hparams = {
    'learning_rate': 5,
    'epochs':        5,
    'train_size':    100, 
    'test_size':     20,
    'n_hidden':      100,
    'n_input':       50,
    'n_output':      50,
    'logint':        1, 
    'print_pred':    0, 
    'plot':          True,
    'embedding':     'rand',
    'rand_seed':     555}

hparams['n_hidden'] = 50
run()
hparams['n_hidden'] = 100
run()
hparams['n_hidden'] = 200
run()


# hidden size 50 the best.
# 
# ## Learning rate

# In[194]:


hparams = {
    'learning_rate': 5,
    'epochs':        5,
    'train_size':    100, 
    'test_size':     20,
    'n_hidden':      50,
    'n_input':       50,
    'n_output':      50,
    'logint':        1, 
    'print_pred':    0, 
    'plot':          True,
    'embedding':     'rand',
    'rand_seed':     555}

hparams['learning_rate'] = 1
run()
hparams['learning_rate'] = 5
run()
hparams['learning_rate'] = 10
run()


# 1 converges quickest
# 
# # Final model

# In[196]:


hparams = {
    'learning_rate': 1,
    'epochs':        5,
    'train_size':    6036, 
    'test_size':     750,
    'n_hidden':      50,
    'n_input':       50,
    'n_output':      50,
    'logint':        1, 
    'print_pred':    10, 
    'plot':          True,
    'embedding':     'rand',
    'rand_seed':     555}
run()


# The final model and the example of prediction are printed. It can be improved a lot if more time is available.
# 
# ## Most frequently right and wrong words

# In[289]:


all_keys = set(list(most_right.keys()) + list(most_wrong.keys()))
total_appearance = {}
for key in all_keys:
    if (key in most_right) and (key in most_wrong):
        total_appearance[key] = most_right[key] + most_wrong[key]
    elif (key in most_right) and (key not in most_wrong):
        total_appearance[key] = most_right[key]
    else:
        total_appearance[key] = most_wrong[key]
most_right = {key:(most_right[key]/total_appearance[key]) for key in most_right}
right_freq = sorted(most_right.items(), key=lambda kv: kv[1], reverse = True)[0:20]
right = [x for x, y in right_freq]
most_wrong = {key:(most_wrong[key]/total_appearance[key]) for key in most_wrong}
wrong_freq = sorted(most_wrong.items(), key=lambda kv: kv[1], reverse = True)[0:20]
wrong = [x for x, y in wrong_freq]
print(most_right)
# print(most_wrong)
print("20 Most frequently right words:", print_list(right))
print("20 Most frequently wrong words:", print_list(wrong))


# Note these are calculated as frequency instead of number of occurrence
# 
# ## Prediction on test set

# In[270]:


test_losses, test_acc, all_pred = run_test(val_input, sample_verbose = 0, embed_type = "rand", return_all_pred = True)
with open("bobsue-data/test.prediction.txt", "w") as f:
    for i in range(len(all_pred)):
        y = '<s> ' + print_list(all_pred[i])
        f.write("%s\n" % y)
        if i < 10: print(y)


# # References
# 
# * [Understanding LSTM Networks -- colah's blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# * [How to use Pre-trained Word Embeddings in PyTorch – Martín Pellarolo – Medium](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)
# * [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
# * [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) - YouTube](https://www.youtube.com/watch?v=WCUNPb-5EYI)
# * [examples/train.py at master · pytorch/examples](https://github.com/pytorch/examples/blob/master/snli/train.py)
# * [examples/main.py at master · pytorch/examples](https://github.com/pytorch/examples/blob/master/word_language_model/main.py)
# * [Creating A Text Generator Using Recurrent Neural Network - Chun’s Machine Learning Page](https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/)
# * [LSTM Neural Network from Scratch | Kaggle](https://www.kaggle.com/navjindervirdee/lstm-neural-network-from-scratch)
# * [lstm/lstm.py at master · nicodjimenez/lstm](https://github.com/nicodjimenez/lstm/blob/master/lstm.py)
# * [LSTMs for Time Series in PyTorch | Jessica Yung](http://www.jessicayung.com/lstms-for-time-series-in-pytorch/)
# * [tutorials/sequence_models_tutorial.py at master · pytorch/tutorials](https://github.com/pytorch/tutorials/blob/master/beginner_source/nlp/sequence_models_tutorial.py)
# * [学习笔记CB012: LSTM 简单实现、完整实现、torch、小说训练word2vec lstm机器人 - OurCoders (我们程序员)](http://ourcoders.com/thread/show/9491/)
# * [Long short-term memory - Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
