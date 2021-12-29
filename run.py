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
    ensemble_ids = list(range(128))
    path = '/opt/shared-disk2/sychou/ensemble'
    epoch = 10

    config = {
        'Section Train Ensemble':{
            'Group Train Ensemble':{
                'Call': train_an_ensemble,
                'Param': {
                    'id': ensemble_ids,
                    'epoch': [epoch],
                    'batch_size': [32],
                    'sel_label': [None], 
                    'base_path': [path],
                },
                'Async':{
                    'gpu_id': [0, 1, 2, 3]
                }
            }
        }
    }

    run(config=config)