import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default='../../data/train.tsv')
    parser.add_argument("--dev", type=str, default='../../data/dev.tsv')
    parser.add_argument("--test", type=str, default='../../data/test.tsv')
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--main_lrate", type=float, default=0.00005)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--pretrained_model", type=str, default="imvladikon/charbert-bert-wiki")  # google/electra-base-discriminator vinai/bertweet-base
    parser.add_argument("--weighting_model", type=str, default="google/electra-base-discriminator")  # imvladikon/charbert-bert-wiki
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--emotions_txt", type=str, default="../../data/emotions.txt")

    # parser.add_argument("--continue_train", type=bool, default=False)
    # parser.add_argument("--continue_train_epoch", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--lambda_loss", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=58097)  # Pycharm debug
    parser.add_argument("--mode", type=str, default='client')

    args = parser.parse_args()
    print(f"RUN:{vars(args)}")
    return args
