# Feature-Space-Augmentation-and-Iterated-Learning
Official implementation for [https://arxiv.org/abs/2405.01705](Long Tail Image Generation Through Feature Space Augmentation and Iterated Learning), accepted in (LXAI CVPR2024)[https://research.latinxinai.org/workshops/cvpr/cvpr.html] as an extended abstract.

### Usage
The tested dataset is available (here)[https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/1.0.0/] 

* `0_img_to_vector.py` : Convert images to latent vectors.
* `1_train_iterated.py` : Run iterated training for the Encoder, Decoder and Classifier.
* `2_fuse.py` : Create class activation maps and fuse long tail images based on their nearest neighbors.
* `3_vec_to_img.py` : Convert fused vectors into new images.
### Disclaimer WIP
The current code is very memory hungry, proceed with caution and start with few images. 
