import os

from train import train_a_model, ModelMgr, Training, RecordMgr
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
    path = '/opt/shared-disk2/sychou/ensemble/new_ensmeble'
    epoch = 30

    training_loop = Training(batch_size=32, epoch=30, base_path=RecordMgr.RESULT_PATH)

    config = {
        'Section Testing':{
            'Group Testing':{
                'Call': train_a_model,
                'Param': {
                    'width': [1024],
                    'id': [0],
                    'model_type': [ModelMgr.FINITE_CNN_MODEL],
                    'classifier_activation': [None], 
                    'layer_num': [5],
                    'kernel_size': [(3, 3)],
                    'conv_block': [1],
                    'is_freeze': [True],
                    'sel_label': [None], 
                    'closed_form': [True], 
                    'float64': [False]
                },
                'Async':{
                    'gpu_id': ['1', '2', '3']
                }
            },
        },
        # 'Section Model New Ensemble W1024':{
        #     'Group Model New Ensemble W1024':{
        #         'Call': train_a_model,
        #         'Param': {
        #             'width': [1024],
        #             'id': ensemble_ids,
        #             'model_type': [ModelMgr.FINITE_CNN_MODEL],
        #             'classifier_activation': [None], 
        #             'layer_num': [5],
        #             'conv_block': [1],
        #             'is_freeze': [True],
        #             'epoch': [epoch],
        #             'batch_size': [32],
        #             'lr': [0.001],
        #             'sel_label': [None], 
        #             'base_path': [os.path.join(path, 'w1024')],
        #         },
        #         'Async':{
        #             'gpu_id': ['0', '2', '3']
        #         }
        #     },
        # },
        # 'Section Model New Ensemble W2048':{
        #     'Group Model New Ensemble W2048':{
        #         'Call': train_a_model,
        #         'Param': {
        #             'width': [2048],
        #             'id': ensemble_ids,
        #             'model_type': [ModelMgr.FINITE_CNN_MODEL],
        #             'classifier_activation': [None], 
        #             'layer_num': [5],
        #             'conv_block': [1],
        #             'is_freeze': [True],
        #             'epoch': [epoch],
        #             'batch_size': [32],
        #             'lr': [0.001],
        #             'sel_label': [None], 
        #             'base_path': [os.path.join(path, 'w2048')],
        #         },
        #         'Async':{
        #             'gpu_id': ['0', '2', '3']
        #         }
        #     }
        # },
        # 'Section Model New Ensemble W4096':{
        #     'Group Model New Ensemble W4096':{
        #         'Call': train_a_model,
        #         'Param': {
        #             'width': [4096],
        #             'id': ensemble_ids,
        #             'model_type': [ModelMgr.FINITE_CNN_MODEL],
        #             'classifier_activation': [None], 
        #             'layer_num': [5],
        #             'conv_block': [1],
        #             'is_freeze': [True],
        #             'epoch': [epoch],
        #             'batch_size': [32],
        #             'lr': [0.001],
        #             'sel_label': [None], 
        #             'base_path': [os.path.join(path, 'w4096')],
        #         },
        #         'Async':{
        #             'gpu_id': ['0', '2', '3']
        #         }
        #     }
        # }
    }

    run(config=config)