from train import train_an_ensemble
from scalablerunner.taskrunner import TaskRunner

def run(config: dict) -> None: 
    """
    A simple function for running specific config.
    """
    tr = TaskRunner(config=config)
    tr.output_log(file_name='logs/taskrunner.log')
    tr.run()

if __name__ == '__main__':
    ensemble_ids = list(range(64))
    path = '/opt/shared-disk2/sychou/ensemble/w1024'
    epoch = 20

    config = {
        'Section Train Ensemble':{
            'Group Train Ensemble':{
                'Call': train_an_ensemble,
                'Param': {
                    'width': [1024],
                    'id': ensemble_ids,
                    'epoch': [epoch],
                    'batch_size': [32],
                    'sel_label': [None], 
                    'base_path': [path],
                },
                'Async':{
                    'gpu_id': ['0', '2', '3']
                }
            }
        }
    }

    run(config=config)