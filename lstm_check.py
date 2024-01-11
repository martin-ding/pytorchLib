import torch
from lstm import RNN, read_vocab
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

device = torch.device('mps')
vocab = read_vocab('vocab.txt')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
label_transform = lambda x: 1 if x == 2 else 0  # 1 neg 2 pos => pos 1 neg 0
rnn = RNN(len(vocab), 100, 256)
rnn.load_state_dict(torch.load('weights.pt', map_location=device))

def evalStr(rnn, str):
    rnn.eval()
    with torch.no_grad():
        text = torch.tensor(text_transform(str))
        pred = torch.round(torch.sigmoid(rnn(text.unsqueeze(-1)))).item()
        print('prediction :', 'negative' if pred == 0.0 else 'positive')

evalStr(rnn, "I was extremely amused at the contents of this movie.")
