
import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Task name for this training'
        ) 
    
    parser.add_argument(
        '--data_root',
        type=str,
        help='Task name for this training'
        ) 

    parser.add_argument(
        '--task',
        default='segmentation',
        type=str
    )
    
    parser.add_argument(
        '--phase',
        default='train',
        type=str
    )
    
    # -----------------------------------------------------------------------------    
    parser.add_argument(
        "--local-rank", 
        type=int)
    
    parser.add_argument(
         '--device', 
         default='0',
         help='device id (i.e. 0 or 0,1 or cpu)')    
    
    # -----------------------------------------------------------------------------
    parser.add_argument(
        '--n_epochs',
        default=500,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--val_interval',
        default=10,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=5*1e-3,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument(
        '--batch_size', default=4, type=int, help='Batch Size')
    
    parser.add_argument(
        '--patch_size', default=(128, 128, 128), type=int, help='Patch Size')
    
    parser.add_argument(
        '--n_seg_classes',
        default=2,
        type=int,
        help="Number of segmentation classes"
    )
    # -----------------------------------------------------------------------------

    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help=
        'Path for resume model.'
    )
    
    parser.add_argument(
        '--pretrain_path',
        default='',
        type=str,
        help=
        'Path for pretrained model.'
    )

    # -----------------------------------------------------------------------------
    

    parser.add_argument(
        '--model',
        default='UNet_baseline',
        type=str)

    args = parser.parse_args()

    return args
