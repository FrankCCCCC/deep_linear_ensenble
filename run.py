from train import train_a_model, ModelMgr
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
    path = '/opt/shared-disk2/sychou/ensemble/model_search'
    epoch = 30

    config = {
        'Section Model Arch Search':{
            'Group Model Arch Search':{
                'Call': train_a_model,
                'Param': {
                    'width': [256, 512, 1024, 2048, 4096],
                    # 'id': ensemble_ids,
                    'id': [None],
                    'model_type': [ModelMgr.FINITE_CNN_MODEL],
                    'classifier_activation': [None], 
                    'layer_num': [5],
                    'conv_block': [1, 2, 3],
                    'is_freeze': [True],
                    'epoch': [epoch],
                    'batch_size': [32],
                    'lr': [0.001],
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