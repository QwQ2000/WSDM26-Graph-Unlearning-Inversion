import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=2, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Unlearn', choices=["Unlearn", "Attack", "Inversion"])
    parser.add_argument('--method', type=str, default='GIF', choices=["GIF", "Retrain", "IF", 'CEU', 'GA'])

    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics", "ogbn-arxiv", "lastfm-asia"])
    parser.add_argument('--unlearn_task', type=str, default='edge', choices=["edge", "node", 'feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.05)

    ########################## training parameters ###########################
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    #parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    #parser.add_argument('--is_retrain', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=True)
    #parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GCN', choices=["SAGE", "GAT", 'MLP', "GCN", "GIN","SGC"])
    parser.add_argument('--target_model_layer', type=int, default=2)
    #parser.add_argument('--train_lr', type=float, default=0.01)
    #parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=100)#5)
    parser.add_argument('--scale', type=int, default=500)#50)
    parser.add_argument('--damp', type=float, default=0.0)

    ########################## Inversion parameters ###########################
    parser.add_argument('--partition_method', type=str, default='metis', choices=["metis", "random"])
    parser.add_argument('--random_part_ratio', type=float, default=0.5, help='shadow/attack nodes ratio for random partitioning')
    parser.add_argument('--metis_parts', type=int, default=2, help='number of parts for metis partitioning')
    parser.add_argument('--metis_shadow_parts', type=int, default=1, help='number of parts for metis partitioning')
    parser.add_argument('--lp_attack_model', type=str, default='SAGE', choices=["SAGE", "GAT", "MLP"])
    parser.add_argument('--attack_method', type=str, default='mia_gnn', choices=["mia_gnn", "transfer_lp", 
                                                                                 "steal_link", "trend_mia", 
                                                                                 "trend_steal", "group_attack"])
    parser.add_argument('--attack_train_neg_ratio', type=float, default=1)
    parser.add_argument('--attack_test_neg_ratio', type=float, default=1)
    parser.add_argument('--num_neighbors', type=int, default=-1)
    parser.add_argument('--ceu_noise_var', type=float, default=0.001)
    parser.add_argument('--ga_neg_alpha', type=float, default=0.5)
    parser.add_argument('--ga_epochs', type=int, default=1)
    parser.add_argument('--trend_k', type=int, default=2)
    # Save/Load intermediate results
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    #parser.add_argument('--rebuild_shadow_attack_data', type=str2bool, default=False)
    #parser.add_argument('--save_shadow_attack_data', type=str2bool, default=True)
    parser.add_argument('--is_gen_unlearn_request', type=str2bool, default=False, help='save unlearn request')
    parser.add_argument('--is_gen_unlearned_probs', type=str2bool, default=False, help='generate probs from unlearned models')
    
    # Save intermediate data for external LP baselines
    parser.add_argument('--export_data', type=str2bool, default=False, help='Export data without running the attack')
    args = vars(parser.parse_args())

    return args
