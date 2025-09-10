import argparse

from src.experiments import TrainExperiment, PredictExperiment


def parse_args():

    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument(
        'data_config', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        'loss_config', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        'model_config', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        'training_config', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42, 
        help='seed for initialization'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_experiment = TrainExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    train_experiment.run()
    predict_experiment = PredictExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    predict_experiment.run()