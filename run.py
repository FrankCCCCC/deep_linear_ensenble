from train import train
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
    
    config = {
        'Section Train Ensemble':{
            'Group Train Ensemble':{
                'Call': train,
                'Param': {
                    'id': [0, 1],
                    'epoch': [10],
                    'batch_size': [32],
                    'sel_label': [None], 
                    'checkpoint_path': [path],
                },
                'Async':{
                    'gpu_id': [0, 1, 2, 3]
                }
            }
        }
    }

    run(config=config)