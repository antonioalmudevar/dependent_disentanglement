import argparse

from src.experiments import CalcMetricsExperiment


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
    experiment = CalcMetricsExperiment(
        **vars(args), wandb_key='ad6dfde6b67458b23b722ca23221f8d82d3cf713')
    experiment.run()