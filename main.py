from data import path
from data import create_dataset
from option import args
from data import util
def main():
    print(path.PATH(args).paths_train_LR)
    X = create_dataset(args, path.PATH(args).paths_train_LR, path.PATH(args).paths_train_HR, 'v')
    print(X)
    print(X[1]) # 列表从0开始
    print(X[0])




if __name__ == '__main__':
    main()