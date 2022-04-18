'''
The overall wrapper
'''
from multiprocessing.sharedctypes import Value
from preprocess import read_dataset, convert_text_to_ids, data_iter, pad_sentences
from utils import Vocab, evaluate, labelMapping
from model import biLSTM
import time
import torch
import torch.nn as nn

# TODO: We first train this model on Google Colab,
# If colab's computational power is not enough,
# we will move it to Google Cloud & add arg parser.
def get_args():
    return 


def main():
    # Loading Data
    train_text = read_dataset('Others/biLSTM/trial_data/train.tsv')
    dev_text = read_dataset('Others/biLSTM/trial_data/dev.tsv')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    # Construct vocabulary from our text data
    word_vocab = Vocab(pad=True, unk=True)
    word_vocab.build(list(zip(*train_text))[0])
    NUM_CLASSES = 28 # This one is hard-coded, maybe pass-in by argparse
                     # Better way would be checking set length
                     # For multiclass, not that straight-forward
                     # being lazy here
    train_data = convert_text_to_ids(train_text, word_vocab, NUM_CLASSES)
    dev_data = convert_text_to_ids(dev_text, word_vocab, NUM_CLASSES)
    # Set model hyperparameters 
    EMBEDDING_DIM = 100
    EMBEDDING_DIR = 'Others/biLSTM/embeddings/glove.twitter.27B/glove.twitter.27B.100d.txt'
    HIDDEN_DIM = 300
    HIDDEN_FC = 512
    NUM_LAYER = 4
    DROPOUT = 0
    lr = 8e-3
    model = biLSTM(vocab=word_vocab, embedding_dim=EMBEDDING_DIM, 
                   embedding_dir=EMBEDDING_DIR, hidden_dim=HIDDEN_DIM,
                   hidden_fc=HIDDEN_FC, num_layer=NUM_LAYER, num_class=NUM_CLASSES,
                   dropout=DROPOUT)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    MAX_EPOCH = 10
    BATCH_SIZE = 32
    LOG_NITER = 50
    EVAL_NITER = 1000
    # Training block
    start_time = time.time()
    train_iter = 0
    train_loss = train_example = train_correct = 0
    best_record = (0, 0) # Best iteration, best accuracy
    model.train()
    for epoch in range(MAX_EPOCH):
        for batch in data_iter(train_data, batch_size=BATCH_SIZE,shuffle=True):
            train_iter += 1
            X, seq_length = pad_sentences(batch[0], word_vocab['<pad>'])
            X = torch.LongTensor(X).to(device)
            Y = torch.FloatTensor(batch[1]).to(device)
            # Forward Pass
            scores = model(X, seq_length) # TODO: Forward length??

            

            loss = loss_function(scores, Y)
            optimizer.zero_grad()
            # Backprop
            loss.backward()
            # Update
            optimizer.step()

            # Compute loss
            train_loss += loss.item() * len(batch[0])
            train_example += len(batch[0])
            # TODO: How many training sample is correctly predicted?
            helper = nn.Sigmoid()
            prediction = (helper(scores) >= 0.5).float()
            num_of_prediction = prediction.size(0)
            for i in range(num_of_prediction):
                train_correct += int(torch.equal(prediction[i], Y[i]))
            # print logs
            if train_iter % LOG_NITER == 0:
                print(f'Epoch {epoch}, iter {train_iter}, training set: '\
                    f'loss={train_loss/train_example:.4f}, ' \
                    f'accuracy={train_correct/train_example:.2f} ({train_correct}/{train_example}), '\
                    f'time={time.time() - start_time:.2f}s')
                train_loss = train_example = train_correct=0
            # Evaluate on dev set
            if train_iter % EVAL_NITER == 0:
                model.eval()
                print(f'Evaluate dev data:')
                with torch.no_grad():
                    dev_accuracy = evaluate(dev_data, model, device)
                    if dev_accuracy > best_record[1]:
                        print(f'  -Update best model at {train_iter}, dev accuracy={dev_accuracy:.4f}')
                        best_record = (train_iter, dev_accuracy)
                #    torch.save('Others/biLSTM/output/good_model.pt')
    # Evaluate on the best one
    # torch.load('Others/biLSTM/output/good_model.pt')
    # evaluate(dev_data, model, device, 
    #         labelMapping.get_mapping(), 
    #         filename='Others/biLSTM/output/good_model_result.txt')


if __name__ == '__main__':
    main()