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


Train the protein design model
------------------------------

    $ python train_test_design_fp16_v13v3best2.py


Fine-tune the protein design model for TadA
-------------------------------------------

    $ python train_test_design_fp16_v13v3best2_tadA.py
    


Fine-tune the ESM2 model for TadA
---------------------------------

    $ python run_predict_pro
    


Generate probability distribution
---------------------------------

    $ python run_predict_proteindesign_v3_tadA.py -i 8e2p.chainA.pdb -o finetune_tada_8e2p
    


Protein sequence generation
---------------------------

    $ python generate_tada_sequences.py
    


Citation
--------

Please cite the following article:

- Ye Yuan, Yang Chen, Rui Liu, Gula Que, Yina Yuan, Haihua Cai, Guipeng Li. An AI-designed adenine base editor. doi:[https://doi.org/10.1101/2024.04.28.591233](https://www.biorxiv.org/content/10.1101/2024.04.28.591233v2.full)


Contact
-------

Author: Dr. Guipeng Li

Email:  guipeng.lee(AT)gmail.com
