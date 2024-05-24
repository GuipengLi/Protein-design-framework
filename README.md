Protein design framework
========================

This repository present the protein design framework described in the [paper](https://www.biorxiv.org/content/10.1101/2024.04.28.591233v2.full). These codes are only available for non-commercial usage.


Requirements
------------

The following python packages are required.

- numpy, pandas
- pytorch
- torch_scatter
- [ESM2](https://github.com/facebookresearch/esm)
- biotite


Step1. Train the protein design model
-------------------------------------

    $ python train_test_design_fp16_v13v3best2.py


Step2. Fine-tune the protein design model for TadA
--------------------------------------------------

Before runing this script, you need to prepare the dataset( using ESMFold to generated pdb files). Then set the path of checkpoint for the pre-tained model from step1. The esmfold_inference.py file is directly from [ESM2](https://github.com/facebookresearch/esm).

    $ python esmfold_inference.py -i seqdump_8e2p_top4791.fa -o esmfold/8e2p_top4791 > seqdump_8e2p_top4791.log
    $ python train_test_design_fp16_v13v3best2_tadA.py
    


Step3. Fine-tune the ESM2 model for TadA
----------------------------------------

Prepare the dataset: Execute PREdata_TadA.py, providing the fasta file as input, and obtain pt files stored within the train_4791 and test_4791 directories.

    $ python PREdata_TadA.py
    

In TadA_train.py, specify the paths for the train_4791 and test_4791 directories. Run the code for training. The training process will retain the optimal model based on the loss on the test set. Multi-GPU training is supported in the training process. Example usage:

    $ CUDA_VISIBLE_DEVICES=0,1 python TadA_train.py
    


Step4. Generate probability distribution
----------------------------------------

Before runing this script, you need to set the path of checkpoint for the fine-tuned model from step2.

    $ python run_predict_proteindesign_v3_tadA.py -i 8e2p.chainA.pdb -o finetune_tada_8e2p
    


Step5. Protein sequence generation
----------------------------------

    $ python generate_tada_sequences.py
    


Citation
--------

Please cite the following article:

- Ye Yuan, Yang Chen, Rui Liu, Gula Que, Yina Yuan, Haihua Cai, Guipeng Li. An AI-designed adenine base editor. doi:[https://doi.org/10.1101/2024.04.28.591233](https://www.biorxiv.org/content/10.1101/2024.04.28.591233v2.full)


Contact
-------

Author: Dr. Guipeng Li

Email:  guipeng.lee(AT)gmail.com
