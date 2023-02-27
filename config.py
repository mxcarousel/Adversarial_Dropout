import argparse


def get_config():
    parser = argparse.ArgumentParser()
    # distribute
    parser.add_argument('-S', '--size', help='size, default=10',
                        dest='size', type=int, default=40)
    parser.add_argument('-SS', '--sample_frac', help='sample_frac',type=float,default=0.1)
    parser.add_argument('-SU', '--speedup', help='speedup ratio, no speedup default=1,1/p,1/sqrt(p)', type=float,
                        default=0)
    # model
    parser.add_argument('-mn', '--model_name', help='model name, default=\'Linear\'',
                        dest='model_name', type=str, default='Logistic')
    parser.add_argument('-A', '--algorithm', help='algorithm type: AMFED, MIFA, FEDAVG, AMFED_Prox, FEDAVG_Prox', default='AMFED',
                        type=str)
    # dropout
    parser.add_argument('--sample_strategy', help='sampling strategy: uar withp', type=str, default='uar')
    parser.add_argument('--dropout', help='drop-out scheme mode: forever, alternate', type=str,default='alternate')
    parser.add_argument('--alpha', help='adversarial drop-out rate', type=float,
                        default=0.1)
    parser.add_argument('--threshold', help='dropout threshold', type=int, default=5)
    parser.add_argument('--repetition', help='no greater than 10 dropout 2', type=int, default=4)
    parser.add_argument('--interval', help='dropout2', type=int, default=100)
   
    # dataset
    parser.add_argument( '--path', help='path of dataset, default=\'../data\'',
                        dest='path', type=str, default='../data')
    parser.add_argument('--dataset_name', help='dataset_name, default=\'MNIST\'',
                        dest='dataset_name', type=str, default='MNIST')
    parser.add_argument('--batch_size', help='batch_size, default=32',
                        dest='batch_size', type=int, default=32)
    parser.add_argument('--iid', help='iid distribution, 1: iid, 2: given class',
                        type=int, default=2)
    parser.add_argument('--class_per_client', help='class_per_client', type=int, default=2)
    parser.add_argument('--n_swap', help='n_swap, default=0',
                        dest='n_swap', type=int, default=None)
    # train
    parser.add_argument('-lr', '--learning_rate', help='learning_rate, default=0.1',
                        dest='lr', type=float, default=0.1)
    parser.add_argument('-mu','--mu', help='Proximal parameter for FedProx', dest='mu',
                        type=float, default=0.1)
    parser.add_argument('--step_decay', help='sqrt/ linear/ constant', type=str, default='sqrt')

    parser.add_argument('-wd', '--weight_decay', help='weight_decay, default=1e-4',
                        dest='weight_decay', type=float, default=1e-3)
    parser.add_argument('-cr', '--communication_round', help='communication round, default=500',
                        dest='communication_round', type=int, default=500)
    parser.add_argument('-ls', '--local_steps', help='local_step_#', type =int, dest='local_steps', 
                        default=0)
    parser.add_argument('-le', '--local_epochs', help = 'local epochs', type=int, default=2)
    parser.add_argument('-ibl', '--imbalanced', help='balanced or not 0: balanced, 1: Gaussian, 2: Zipf', 
                        dest='imbalanced', type=int, default=1)
    parser.add_argument('-g', '--gamma', help='gamma, default=1 constant step',
                        dest='gamma', type=float, default=1)
    parser.add_argument('--seed', help='seed, default=777',
                        dest='seed', type=int, default=777)
    # device
    parser.add_argument('-G', '--gpu', help='use gpu for train default True',
                        dest='gpu', type=bool, default=True)
 
    args = parser.parse_args()
    return args.__dict__
