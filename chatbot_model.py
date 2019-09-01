
import itertools
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import re as r
import unicodedata
import os




USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
#corpus_name = "cornell movie-dialogs corpus"
#corpus = os.path.join("data", corpus_name)

# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
save_dir = os.path.join("save")
# Creating a dictionary that maps each line and its id
id2line = {}
for l in lines:
    _l = l.split(' +++$+++ ')
    if len(_l) == 5:
        id2line[_l[0]] = _l[4]
        
que,ans=[],[]
conv_ids = []
for c in conversations[:-1]:
    _a = c.split(' +++$+++ ')[-1][2:-2]
    conv_ids.append(_a.split("', '"))
    
for c in conv_ids:
    for i in range(len(c)-1):
        que.append(id2line[c[i]])
        ans.append(id2line[c[i+1]])


 


with open("data2.txt",'r') as f:
   c=f.read()
a=c.split("@ ")

for i in a[1:]:
    i=r.sub("  -","- -",i).split("- -")
    for x in range(len(i)-1):
        que.append(i[x])
        ans.append(i[x+1])
        
MAX_LENGTH=10
def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize('NFD',s) if unicodedata.category(c)!='Mn')

def p_process_string(s):
    s=unicodeToAscii(s.lower().strip())
    
    s=r.sub(r"([.!?])",r" \1 ",s)
    s=r.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s=r.sub(r"\s+",r" ",s).strip()
    return s


for i in range(len(que)):
    que[i]=p_process_string(que[i])
    ans[i]=p_process_string(ans[i])



PAD_token = 0
SOS_token = 1
EOS_token = 2



class wordprocess:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}

        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3
        
    def addsent(self,sent):
        for w in sent.split(' '):  self.addword(w)
        
    def addword(self,word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words +=1
        else:
            self.word2count[word] += 1
            
    def remword(self,min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
       # print('keep_words{}/{}={:.4f}'.format(len(keep_words),  len(self.word2index),len(keep_words) / len(self.word2index)))
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.index2word = {PAD_token:"PAD",SOS_token:"SOS",EOS_token:"EOS"}
        self.num_words = 3
        
        for w in keep_words:
            self.addword(w)


wp=wordprocess("movie_dialogues")

for i in range(len(que)):
    wp.addsent(que[i])
    wp.addsent(ans[i])
print("counted words : ",wp.num_words)

dic = wp.word2index 
  
# now filter the sentences which are too long
    
min_count = 3
def sentence_selector(wp,que,ans,min_count):
    wp.remword(min_count)

    pairs=[]
    que1,ans1=[],[]
    for k in range(len(que)):
        flag,flag1=True,True
        for k1 in que[k].split(" "):
            if k1 not in wp.word2index:
                flag = False
                break

        
        for k1 in ans[k].split(" "):
            if k1 not in wp.word2index:
                flag1=False
                break
        if flag and flag1:
            que1.append(que[k])
            ans1.append(ans[k])
           # print("que1:",que[k])
    pairs.append(que1)
    pairs.append(ans1)
    return pairs
pairs=sentence_selector(wp,que,ans,min_count)
##########################################################################################################################################


def sent2index(wp,sent):
    return [wp.word2index[w] for w in sent.split(" ")] + [EOS_token]

def zeroPadding(l,fillvalue=0):
    return list(itertools.zip_longest(*l,fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    m=[]
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(l, wp):
    indexes_batch = [sent2index(wp, sent) for sent in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar  = torch.LongTensor(padList)
    return padVar , lengths

def outputVar(l, wp):
    indexes_batch = [sent2index(wp,sent) for sent in l] 
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar , mask ,max_target_len


def batch2TrainData(wp,pairs):
    input_batch,output_batch = pairs[0],pairs[1]
    inp,lengths = inputVar(input_batch,wp)
    output,mask,max_target_len = outputVar(output_batch,wp)
    return inp,lengths,output,mask,max_target_len



count_words=wp.num_words
len_dic=len(wp.word2index)
print("length of dictionary:  ",len_dic)
#wp.word2index['eber']=27949
#wp.word2index['blowdryer']=27950

###########################################################    DEFINING   MODEL


class Encoder(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(Encoder,self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout ),bidirectional=True)
                
    def forward(self,input_seq,input_lens,hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded,input_lens,enforce_sorted=False)
        outputs, hidden = self.gru(packed,hidden)
        
        outputs,_ =  nn.utils.rnn.pad_packed_sequence(outputs)
        
        outputs = outputs[:,:,:self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs,hidden




    
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)
    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)    
  
    


    
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()  
        self.attn_model=attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size*2 , hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn( hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden    



#####################################################################################################################################

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()    
 
###############################################    ########################################################################################

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0
    print("s1")
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    print("s2")
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

# USING TEACHER FORCING METHOD
    for t in range(max_target_len):
        
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_input = target_variable[t].view(1, -1)
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal
    
    loss.backward()

    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


##############################################################
def trainIters(model_name, wp, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename ):

    a,b,training_batches=[],[],[]
    for _ in range(n_iteration):
        pair=[]
        for i in range(batch_size):            
            a.append(pairs[0][i])
            b.append(pairs[1][i])
        pair.append(a)
        pair.append(b)
        training_batches.append(batch2TrainData(wp,pair))
       
           
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
             directory = os.path.join(save_dir, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
             if not os.path.exists(directory):
                os.mkdirs(directory)
                torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': wp.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))
    
    
######################################################################
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores    
    
 ####################################################
def evaluate(encoder, decoder, searcher, wp, sentence, max_length=MAX_LENGTH):
    indexes_batch = [sent2index(wp, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [wp.index2word[token.item()] for token in tokens]
    return decoded_words



def evaluateInput(encoder, decoder, searcher, wp):
    input_sentence = ''
    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            input_sentence = p_process_string(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, wp, input_sentence)
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")




##################################################################
# Configure models


model_name = 'cb_model'
attn_model='dot'
hidden_size = 128
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 128

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    wp.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(wp.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models

encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, wp.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')





#############################################################################




# Configure training/optimization

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.01
decoder_learning_ratio = 5.0
n_iteration = 8
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()


# Run training iterations
corpus_name="cornell-corpus" 
print("Starting Training!")
trainIters(model_name, wp, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)
print("training stop")





#################################################################################




# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

#evaluateInput(encoder, decoder, searcher, wp)


#problem of eber key in word2index



   