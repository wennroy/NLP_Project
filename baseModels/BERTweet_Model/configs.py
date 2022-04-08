import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default='../../data/goemotions_train.csv')
    parser.add_argument("--dev", type=str, default='../../data/goemotions_dev.csv')
    parser.add_argument("--test", type=str, default='../../data/goemotions_test.csv')
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--main_lrate", type=float, default=0.00001)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--pretrained_model", type=str, default="vinai/bertweet-base")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--emotions_txt", type=str, default="../../data/emotions.txt")

    parser.add_argument("--port", type=int, default=58097)  # Pycharm debug
    parser.add_argument("--mode", type=str, default='client')

    args = parser.parse_args()
    print(f"RUN:{vars(args)}")
    return args