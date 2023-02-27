from torch.utils.tensorboard import SummaryWriter


from utils import np,init_deterministic
from function_utils import dataset_init,worker_init,server_init,train
from config import get_config

def run(config,communication_round, model_name, dataset_name, class_per_client, size, repetition, interval,
        batch_size, lr, weight_decay, gpu, iid, step_decay,speedup,
        seed, alpha, imbalanced, dropout, local_steps, local_epochs, sample_frac,
        n_swap, path, mu, algorithm, **kwargs):
    tb = SummaryWriter(comment=f"seed_{seed}_model_{model_name}_dataset_{dataset_name}_bs_{batch_size}_lr_{lr}_size_{size}_alg_{algorithm}_sample_{sample_frac:g}_alpha_{alpha:g}_le_{local_epochs}_mu_{mu:g}_imbalan_{imbalanced}_iid_{iid}_sd_{step_decay}_do_{dropout}_rep_{repetition}_inter_{interval}_class_{class_per_client}_speedup_{speedup:g}")
    init_deterministic(seed)    
    
    dataset_class, temp_train_loader, temp_test_loader = dataset_init(
                                                        iid = iid,
                                                        dataset_name = dataset_name,
                                                        num_clients = size,
                                                        class_per_client = class_per_client,
                                                        imbalance = imbalanced,
                                                        batch_size=batch_size,
                                                        seed = seed,
                                                        path = path)

    worker_list, input_size, classes, weight, stat, num_step = worker_init(size, dataset_class,\
                                                                             model_name, weight_decay,\
                                                                                lr, gpu,iid)
    server = server_init(model_name, input_size, classes, algorithm, weight, sample_frac)
    print(f"| num_step: {num_step}")
    
    baseline_frac = [1 if i<config['threshold'] else sample_frac for i in range(communication_round)]

    """The following statement defines the local-step as 2*the smallest number of batches."""
    if local_steps == 0:
        config['local_steps'] = np.min(local_epochs * np.array(num_step))
    
    train(config, baseline_frac,server,worker_list,temp_train_loader,temp_test_loader, tb,stat)

    
if __name__ == '__main__':
    config = get_config()
    print(config)
    run(config,**config)