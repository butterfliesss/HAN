""" functions, layers, and architecture of HiGRU """
import numpy as np
import math, copy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as weight_init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import Const
from bert_utils import *
from bert_serving.client import BertClient


class GRUencoder(nn.Module):
    def __init__(self, d_emb, d_out, num_layers, bidirectional):
        super(GRUencoder, self).__init__()
        self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
                            bidirectional=bidirectional, num_layers=num_layers)

    def forward(self, sent, sent_lens):
        """
        :param sent: torch tensor, batch_size x len x dim
        :param sent_lens: numpy tensor, batch_size x 1
        :return:
        """
        device = sent.device
        # seq_len x batch_size x d_rnn_in
        sent_embs = sent.transpose(0, 1)

        # sort by length
        s_lens, idx_sort = np.sort(sent_lens)[::-1].copy(), np.argsort(-sent_lens)
        # s_lens = s_lens.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda(device)
        s_embs = sent_embs.index_select(1, Variable(idx_sort))

        # padding
        sent_packed = pack_padded_sequence(s_embs, s_lens)
        sent_output = self.gru(sent_packed)[0]
        sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

        # unsort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
        sent_output = sent_output.index_select(1, Variable(idx_unsort))

        # batch x seq_len x 2*d_out
        output = sent_output.transpose(0, 1)
        return output


# selfattention
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.5):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1))  # batch, seq_len, seq_len
        attn = attn * (k.size(-1) ** -0.5)
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -1e10)  # mask为1的地方用value填充
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)  # batch, seq_len, dim
        return output, attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(model_dim, model_dim), 4)
        self.dot_product_attention = SelfAttention(dropout)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, key, value, query, attn_mask=None):
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        if attn_mask is not None:
            #attn_mask = attn_mask.repeat(num_heads, 1, 1)
            attn_mask = attn_mask.unsqueeze(1)
        query, key, value = \
            [l(x).view(batch_size, -1, num_heads, dim_per_head).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # scaled dot product attention
        context, attention = self.dot_product_attention(
            query, key, value, attn_mask)
        # concat heads
        context = context.transpose(1, 2).contiguous()\
        .view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linears[-1](context)
        # dropout
        # output = self.dropout(output)
        # output = output

        return output, attention


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        L = x.size(0)
        pos_emb = self.pe[:, :L]
        return pos_emb

class myPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        """初始化。
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(myPositionalEncoding, self).__init__()
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000.0, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding).float()
        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat([pad_row, position_encoding], dim=0)
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。
        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。
        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        #input_len = torch.from_numpy(input_len)
        # for len in input_len:
        #     print(len)
        input_len = input_len.cuda()
        max_len = torch.max(input_len)
        #print((max_len.item()-input_len.item()) * [0])

        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos =tensor([list(range(1, len + 1)) + [0] * (max_len - len).item() for len in input_len])#.cuda()
        #print(input_pos.size())
        return self.position_encoding(input_pos)

class HAN_BERT(nn.Module):
    def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding):
        super(HAN_BERT, self).__init__()

        self.max_length = worddict.max_length
        self.max_dialog = worddict.max_dialog
        self.d_h2 = d_h2
        self.d_h1 = d_h1

        self.word2index = worddict.word2index
        self.index2word = worddict.index2word

        # load word2vec
        self.embeddings = embedding

        self.use_bert = True
        self.use_bert_weight = True
        self.use_bert_gamma = False
        self.finetune_bert = False
        if self.use_bert and self.use_bert_weight:
            num_bert_layers = 1
            self.logits_bert_layers = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(1, num_bert_layers)))
            if self.use_bert_gamma:
                self.gamma_bert_layers = nn.Parameter(nn.init.constant_(torch.Tensor(1, 1), 1.))

        if self.use_bert:
            from pytorch_pretrained_bert import BertTokenizer
            from pytorch_pretrained_bert.modeling import BertModel
            print('[ Using pretrained BERT features ]')
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
            self.bert_model = BertModel.from_pretrained('bert-large-uncased')#.cuda(self.device)
            if not self.finetune_bert:
                print('[ Fix BERT layers ]')
                self.bert_model.eval()
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            else:
                print('[ Finetune BERT layers ]')
        else:
            self.bert_tokenizer = None
            self.bert_model = None

        self.dropout = nn.Dropout(0.5) #origin 0.5

        self.weight_W_word = nn.Parameter(torch.Tensor(d_h1, d_h1)) #全部改为1
        # self.bias_word = nn.Parameter(torch.Tensor(d_h1, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(d_h1, 1))
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

        self.bias_word = nn.Parameter(torch.Tensor(d_h1, 1))

        self.dropout_in = nn.Dropout(0.5)
        self.encoder_layers = nn.ModuleList([
            MultiHeadAttention(d_h2, num_heads=32, dropout=0.5) for _ in range(1)
        ])
        self.pos_embedding = PositionalEncoding(d_h2)
        # self.mypos_embedding = myPositionalEncoding(d_h2)
        
        self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=True)
        # self.init_weights()

        self.d_input = 2 * d_h2

        self.output1 = nn.Sequential(
            nn.Linear(self.d_input, d_h2),
            nn.Tanh()
        )
        self.dropout_mid = nn.Dropout(0.5)
        self.num_classes = emodict.n_words
        self.classifier = nn.Sequential(
            nn.Linear(d_h2, d_fc),
            nn.Dropout(0.5),
            nn.Linear(d_fc, self.num_classes)
        )

    # def init_weights_word(self):
    #     for w in self.word_gru.parameters():
    #         if w.dim() > 1:
    #             weight_init.orthogonal_(w.data)

    # def init_weights(self):
    #     for w in self.contenc.parameters():
    #         if w.dim() > 1:
    #             weight_init.orthogonal_(w.data)

    def reverse_seq(self, s_embed):  # batch x d_h1;
        s_embed_rev = torch.flip(s_embed, [0])
        return s_embed_rev

    # functions to accoplish attention
    def batch_matmul_bias(self, seq, weight, bias, nonlinearity=''): # seqlen, batch, dim
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            if (nonlinearity == 'tanh'):
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if (s is None):
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s

    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if (nonlinearity == 'tanh'):
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s.squeeze(-1)

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if (attn_vectors is None):
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)

    def ids_to_word(self, ids):
        return [self.index2word[int(x)] for x in ids if x != 0]

    def get_sent_words(self, sent):
        sent_words = re.split("\\s+", sent)
        return sent_words

    def forward(self, sents, lengths):
        # sents为feat：一个对话的句子长度（个数）， max_seq_len(每个句子的token padding个数)
        # lengths为一个对话中每句话的实际token个数，维度为一个对话的句子个数
        """
        :param sents: batch x seq_len
        :param lengths: 1 x batch
        :return:
        """
        
        device = sents.device
        input_lens = torch.tensor([sents.size(0)])#.unsqueeze(1)

        if len(sents.size()) < 2:
            sents = sents.unsqueeze(0)
        
        # for bert word embeddings
        sent_bert = []
        sents_text = [" ".join(self.ids_to_word(sent)) for sent in sents]
        for i, sent in enumerate(sents_text):
            sent_words = self.get_sent_words(sent)
            bert_sent_features = convert_text_to_bert_features(sent_words, self.bert_tokenizer, 500, 250)
            sent_bert.append(bert_sent_features)
        
        batch_size = len(sents_text)
        if self.use_bert:
            with torch.set_grad_enabled(False):
                layer_indexes = [22]

                max_d_len = lengths.max().item()
                max_bert_d_num_chunks = max([len(ex_bert_d) for ex_bert_d in sent_bert])
                max_bert_d_len = max([len(bert_d.input_ids) for ex_bert_d in sent_bert for bert_d in ex_bert_d])
                bert_xd = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
                bert_xd_mask = torch.LongTensor(batch_size, max_bert_d_num_chunks, max_bert_d_len).fill_(0)
                for i, ex_bert_d in enumerate(sent_bert):
                    for j, bert_d in enumerate(ex_bert_d):
                        bert_xd[i, j, :len(bert_d.input_ids)].copy_(torch.LongTensor(bert_d.input_ids))
                        bert_xd_mask[i, j, :len(bert_d.input_mask)].copy_(torch.LongTensor(bert_d.input_mask))
                device = sents.device
                bert_xd = bert_xd.cuda(device)
                bert_xd_mask = bert_xd_mask.cuda(device)
                all_encoder_layers, _ = self.bert_model(bert_xd.view(-1, bert_xd.size(-1)), token_type_ids=None,
                                                        attention_mask=bert_xd_mask.view(-1, bert_xd_mask.size(-1)))
                torch.cuda.empty_cache()
                all_encoder_layers = torch.stack([x.view(bert_xd.shape + (-1,)) for x in all_encoder_layers],
                                                 0).detach()
                all_encoder_layers = all_encoder_layers[layer_indexes]
                bert_context_f = extract_bert_hidden_states(all_encoder_layers, max_d_len, sent_bert,
                                                            weighted_avg=self.use_bert_weight)
                torch.cuda.empty_cache()

        if self.use_bert:
            context_bert = bert_context_f
            if not self.finetune_bert:
                assert context_bert.requires_grad == False
            if self.use_bert_weight:
                weights_bert_layers = torch.softmax(self.logits_bert_layers, dim=-1)
                if self.use_bert_gamma:
                    weights_bert_layers = weights_bert_layers * self.gamma_bert_layers

                context_bert = torch.mm(weights_bert_layers, context_bert.view(context_bert.size(0), -1)).view(
                    context_bert.shape[1:])

        w_embed = context_bert#.cuda(device)
        w_embed = w_embed.transpose(0, 1).contiguous() # max_seq_len, batch, dim
        output_word = w_embed

        word_squish = self.batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = self.batch_matmul(word_squish, self.weight_proj_word)
        word_attn = word_attn.transpose(1, 0) # batch, seq_len

        attn_mask = torch.zeros(word_attn.size(0), word_attn.size(1)).cuda(device)
        for i in range(word_attn.size(0)):
            attn_mask[i, lengths[i]:] = 1
        attn_mask=attn_mask.eq(1)
        word_attn = word_attn.masked_fill(attn_mask, -1e10)
        word_attn_norm = F.softmax(word_attn, dim=-1)
        word_attn_vectors = self.attention_mul(output_word, word_attn_norm.transpose(1, 0))
        s_embed = word_attn_vectors #batch, dim
        s_embed = self.dropout_in(s_embed)  # batch x d_h1
        # for individual utterance embedding

        s_context = self.contenc(s_embed.unsqueeze(1))[0]
        s_context = s_context.transpose(0, 1).contiguous()
        s_lcont, s_rcont = s_context.chunk(2, -1)

        SA_lcont = s_lcont + s_embed
        SA_rcont = s_rcont + s_embed

        #position emb
        pos_emb = self.pos_embedding(s_embed)
       
        # pos_emb_r = self.reverse_seq(pos_emb.squeeze(0))
        # pos_emb_r = pos_emb_r.unsqueeze(0)
        
        pos_SA_lcont = SA_lcont + pos_emb #mypos_emb
        pos_SA_rcont = SA_rcont + pos_emb #mypos_emb#_r

        # attention_fs = []
        # attention_rs = []
        for encoder in self.encoder_layers: #contain residual
            # SA_lcont, attention_f = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont
            # SA_rcont, attention_r = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont
            SA_lcont = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont #pos_SA_lcont #
            SA_rcont = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont #pos_SA_rcont #
            # attention_fs.append(attention_f)
            # attention_rs.append(attention_r)

        Combined = [SA_lcont, SA_rcont]
        Combined = torch.cat(Combined, dim=-1) #1, batch, 2d_h2

        output1 = self.output1(Combined.squeeze(0))
        output1 = self.dropout_mid(output1)

        output = self.classifier(output1)  # batch, d_fc
        pred_scores = F.log_softmax(output, dim=1)

        return pred_scores


class HAN_Word2Vec(nn.Module):
    def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding):
        super(HAN_Word2Vec, self).__init__()
        
        self.max_length = worddict.max_length
        self.max_dialog = worddict.max_dialog
        self.d_h2 = d_h2
        self.d_h1 = d_h1

        self.word2index = worddict.word2index
        self.index2word = worddict.index2word

        # load word2vec
        self.embeddings = embedding
        self.dropout = nn.Dropout(0.5)
        self.utt_gru = GRUencoder(d_word_vec, d_h1//2, num_layers=1, bidirectional=True)

        # for iemocap
        # self.weight_W_word = nn.Parameter(torch.Tensor(d_h1, d_h1))
        # nn.init.xavier_uniform_(self.weight_W_word, gain=1) #ok
        # self.weight_proj_word = nn.Parameter(torch.Tensor(d_h1, 1))
        # nn.init.xavier_uniform_(self.weight_proj_word, gain=1)

        # for meld
        self.weight_W_word = nn.Parameter(torch.Tensor(d_h1, d_h1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(d_h1, 1))
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

        self.bias_word = nn.Parameter(torch.Tensor(d_h1, 1))

        self.dropout_in = nn.Dropout(0.5)
        self.encoder_layers = nn.ModuleList([
            MultiHeadAttention(d_h2, num_heads=1, dropout=0.5) for _ in range(1)
        ])
        self.pos_embedding = PositionalEncoding(d_h2)
        # self.mypos_embedding = myPositionalEncoding(d_h2)
        
        self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=True)

        self.d_input = 2 * d_h2
        self.output1 = nn.Sequential(
            nn.Linear(self.d_input, d_h2),
            nn.Tanh()
        )
        self.dropout_mid = nn.Dropout(0.5)
        self.num_classes = emodict.n_words
        self.classifier = nn.Sequential(
            nn.Linear(d_h2, d_fc),
            nn.Dropout(0.5),
            nn.Linear(d_fc, self.num_classes)
        )

    # def reverse_seq(self, s_embed):  # batch x d_h1;
    #     s_embed_rev = torch.flip(s_embed, [0])
    #     return s_embed_rev

    # functions to accoplish attention
    def batch_matmul_bias(self, seq, weight, bias, nonlinearity=''): # seqlen, batch, dim
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            if (nonlinearity == 'tanh'):
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if (s is None):
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s

    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if (nonlinearity == 'tanh'):
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s.squeeze(-1)

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if (attn_vectors is None):
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)

    def forward(self, sents, lengths):
        # sents为feat：一个对话的句子长度（个数）， max_seq_len(每个句子的token padding个数)
        # lengths为一个对话中每句话的实际token个数，维度为一个对话的句子个数
        """
        :param sents: batch x seq_len
        :param lengths: 1 x batch
        :return:
        """
        device = sents.device

        if len(sents.size()) < 2:
            sents = sents.unsqueeze(0)
        
        w_embed = self.embeddings(sents)
        w_embed = self.utt_gru(w_embed, lengths)
        w_embed = w_embed.transpose(0, 1).contiguous() 
        output_word = w_embed # max_seq_len, batch, dim
        
        word_squish = self.batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = self.batch_matmul(word_squish, self.weight_proj_word)
        word_attn = word_attn.transpose(1, 0) # batch, seq_len
        attn_mask = torch.zeros(word_attn.size(0), word_attn.size(1)).cuda(device)
        for i in range(word_attn.size(0)):
            attn_mask[i, lengths[i]:] = 1
        attn_mask=attn_mask.eq(1)
        word_attn = word_attn.masked_fill(attn_mask, -1e10)
        word_attn_norm = F.softmax(word_attn, dim=-1)
        word_attn_vectors = self.attention_mul(output_word, word_attn_norm.transpose(1, 0))
        s_embed = word_attn_vectors #batch, dim
        s_embed = self.dropout_in(s_embed)  # batch x d_h1
        # for individual utterance embedding

        s_context = self.contenc(s_embed.unsqueeze(1))[0]
        s_context = s_context.transpose(0, 1).contiguous()
        s_lcont, s_rcont = s_context.chunk(2, -1)

        SA_lcont = s_lcont + s_embed
        SA_rcont = s_rcont + s_embed

        #position emb
        pos_emb = self.pos_embedding(s_embed)

        # pos_emb_r = self.reverse_seq(pos_emb.squeeze(0))
        # pos_emb_r = pos_emb_r.unsqueeze(0)

        pos_SA_lcont = SA_lcont + pos_emb
        pos_SA_rcont = SA_rcont + pos_emb#_r

        # attention_fs = []
        # attention_rs = []
        for encoder in self.encoder_layers: #contain residual
            # SA_lcont, attention_f = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont
            # SA_rcont, attention_r = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont
            SA_lcont = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont # pos_SA_lcont #
            SA_rcont = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont # pos_SA_rcont #
            # attention_fs.append(attention_f)
            # attention_rs.append(attention_r)

        Combined = [SA_lcont, SA_rcont]
        Combined = torch.cat(Combined, dim=-1) #1, batch, 2d_h2

        output1 = self.output1(Combined.squeeze(0))
        output1 = self.dropout_mid(output1)

        output = self.classifier(output1)  # batch, d_fc
        pred_scores = F.log_softmax(output, dim=1)

        return pred_scores


class HAN_BERT_Serivce(nn.Module):
    def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding):
        super(HAN_BERT, self).__init__()

        self.max_length = worddict.max_length
        self.max_dialog = worddict.max_dialog
        self.d_h2 = d_h2
        self.d_h1 = d_h1

        self.word2index = worddict.word2index
        self.index2word = worddict.index2word

        # load word2vec
        self.embeddings = embedding
        self.dropout = nn.Dropout(0.5) #origin 0.5

        self.weight_W_word = nn.Parameter(torch.Tensor(d_h1, d_h1)) #全部改为1
        # self.bias_word = nn.Parameter(torch.Tensor(d_h1, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(d_h1, 1))
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1, 0.1)

        self.bias_word = nn.Parameter(torch.Tensor(d_h1, 1))

        self.dropout_in = nn.Dropout(0.5)
        self.encoder_layers = nn.ModuleList([
            MultiHeadAttention(d_h2, num_heads=32, dropout=0.5) for _ in range(1)
        ])
        self.pos_embedding = PositionalEncoding(d_h2)
        # self.mypos_embedding = myPositionalEncoding(d_h2)
        
        self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=True)
        # self.init_weights()

        self.d_input = 2 * d_h2

        self.output1 = nn.Sequential(
            nn.Linear(self.d_input, d_h2),
            nn.Tanh()
        )
        self.dropout_mid = nn.Dropout(0.5)
        self.num_classes = emodict.n_words
        self.classifier = nn.Sequential(
            nn.Linear(d_h2, d_fc),
            nn.Dropout(0.5),
            nn.Linear(d_fc, self.num_classes)
        )

    def reverse_seq(self, s_embed):  # batch x d_h1;
        s_embed_rev = torch.flip(s_embed, [0])
        return s_embed_rev

    # functions to accoplish attention
    def batch_matmul_bias(self, seq, weight, bias, nonlinearity=''): # seqlen, batch, dim
        s = None
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
            if (nonlinearity == 'tanh'):
                _s_bias = torch.tanh(_s_bias)
            _s_bias = _s_bias.unsqueeze(0)
            if (s is None):
                s = _s_bias
            else:
                s = torch.cat((s, _s_bias), 0)
        return s

    def batch_matmul(self, seq, weight, nonlinearity=''):
        s = None
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], weight)
            if (nonlinearity == 'tanh'):
                _s = torch.tanh(_s)
            _s = _s.unsqueeze(0)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        return s.squeeze(-1)

    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = None
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            if (attn_vectors is None):
                attn_vectors = h_i
            else:
                attn_vectors = torch.cat((attn_vectors, h_i), 0)
        return torch.sum(attn_vectors, 0)

    def ids_to_word(self, ids):
        return [self.index2word[int(x)] for x in ids if x != 0]

    def get_sent_words(self, sent):
        sent_words = re.split("\\s+", sent)
        return sent_words

    def forward(self, sents, lengths):
        # sents为feat：一个对话的句子长度（个数）， max_seq_len(每个句子的token padding个数)
        # lengths为一个对话中每句话的实际token个数，维度为一个对话的句子个数
        """
        :param sents: batch x seq_len
        :param lengths: 1 x batch
        :return:
        """
        
        device = sents.device
        input_lens = torch.tensor([sents.size(0)])#.unsqueeze(1)

        if len(sents.size()) < 2:
            sents = sents.unsqueeze(0)
        
        # for bert word embeddings
        sents_text = [" ".join(self.ids_to_word(sent)) for sent in sents]

        bc = BertClient()
        w_embed = bc.encode(sents_text)  # np.array()类型
        w_embed_list = w_embed.tolist()
        # print(w_embed.shape) #batch, max_seq_len(设为102， 包括[CLS], [SEP]), dim
        # w_embed = torch.from_numpy(w_embed).cuda(device)  #batch, max_seq_len, dim
        batch_size, max_seq_len, bert_dim = w_embed.shape #max_seq_len NONE,自动设置,包括两个特殊符号
        #print(max_seq_len)
        for i in range(batch_size):
            while [0 for _ in range(1024)] in w_embed_list[i]:
                w_embed_list[i].remove([0 for _ in range(1024)])  # 对了
        # print(w_embed_list[0])
        for i in range(batch_size):
            del w_embed_list[i][0]
            del w_embed_list[i][-1]
        # print(w_embed_list)
        lens_all = np.array([len(w_embed_list[i]) for i in range(batch_size)])
        #print(lens_all)#每个句子实际长度
        for i in range(batch_size):
            lens_single = len(w_embed_list[i])
            for _ in range(max_seq_len - 2 - lens_single):
                w_embed_list[i].append([0 for _ in range(1024)])
        w_embed_new = np.array(w_embed_list)
        #print(w_embed_new.shape)#batch, max_seq_len-2(100), dim
        w_embed = torch.from_numpy(w_embed_new).float().cuda(device)
        # print(w_embed.size()) #batch, max_seq_len, dim

        w_embed = w_embed.transpose(0, 1).contiguous()  # batch_first=False
        output_word = w_embed

        word_squish = self.batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')
        word_attn = self.batch_matmul(word_squish, self.weight_proj_word)
        word_attn = word_attn.transpose(1, 0) # batch, seq_len

        attn_mask = torch.zeros(word_attn.size(0), word_attn.size(1)).cuda(device)
        for i in range(word_attn.size(0)):
            attn_mask[i, lengths[i]:] = 1
        attn_mask=attn_mask.eq(1)
        word_attn = word_attn.masked_fill(attn_mask, -1e10)
        word_attn_norm = F.softmax(word_attn, dim=-1)
        word_attn_vectors = self.attention_mul(output_word, word_attn_norm.transpose(1, 0))
        s_embed = word_attn_vectors #batch, dim
        s_embed = self.dropout_in(s_embed)  # batch x d_h1
        # for individual utterance embedding

        s_context = self.contenc(s_embed.unsqueeze(1))[0]
        s_context = s_context.transpose(0, 1).contiguous()
        s_lcont, s_rcont = s_context.chunk(2, -1)

        SA_lcont = s_lcont + s_embed
        SA_rcont = s_rcont + s_embed

        #position emb
        pos_emb = self.pos_embedding(s_embed)
        # print('pos_emb', pos_emb)
       
        # pos_emb_r = self.reverse_seq(pos_emb.squeeze(0))
        # pos_emb_r = pos_emb_r.unsqueeze(0)

        pos_SA_lcont = SA_lcont + pos_emb #mypos_emb
        pos_SA_rcont = SA_rcont + pos_emb #mypos_emb#_r

        # attention_fs = []
        # attention_rs = []
        for encoder in self.encoder_layers: #contain residual
            # SA_lcont, attention_f = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont
            # SA_rcont, attention_r = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont
            SA_lcont = encoder(pos_SA_lcont, pos_SA_lcont, pos_SA_lcont)[0] + SA_lcont #pos_SA_lcont #
            SA_rcont = encoder(pos_SA_rcont, pos_SA_rcont, pos_SA_rcont)[0] + SA_rcont #pos_SA_rcont #
            # attention_fs.append(attention_f)
            # attention_rs.append(attention_r)

        Combined = [SA_lcont, SA_rcont]
        Combined = torch.cat(Combined, dim=-1) #1, batch, 2d_h2

        output1 = self.output1(Combined.squeeze(0))
        output1 = self.dropout_mid(output1)

        output = self.classifier(output1)  # batch, d_fc
        pred_scores = F.log_softmax(output, dim=1)

        return pred_scores