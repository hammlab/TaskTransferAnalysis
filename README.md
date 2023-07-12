# Analysis of Task Transferability in Large Pre-trained Classifiers
Transfer learning transfers the knowledge acquired by a model from a source task to multiple downstream target tasks with minimal fine-tuning. The success of transfer learning at improving performance, especially with the use of large pre-trained models has made transfer learning an essential tool in the machine learning toolbox. However, the conditions under which the performance is transferable to downstream tasks are not understood very well. In this work, we analyze the transfer of performance for classification tasks, when only the last linear layer of the source model is fine-tuned on the target task. We propose a novel Task Transfer Analysis approach that transforms the source distribution (and classifier) by changing the class prior distribution, label, and feature spaces to produce a new source distribution (and classifier) and allows us to relate the loss of the downstream task (i.e., transferability) to that of the source task. Concretely, our bound explains transferability in terms of the Wasserstein distance between the transformed source and downstream task's distribution, conditional entropy between the label distributions of the two tasks, and weighted loss of the source classifier on the source task. Moreover, we propose an optimization problem for learning the transforms of the source task to minimize the upper bound on transferability. We perform a large-scale empirical study by using state-of-the-art pre-trained models and demonstrate the effectiveness of our bound and optimization at predicting transferability. The results of our experiments demonstrate how factors such as task relatedness, pretraining method, and model architecture affect transferability.

<hr>

This repository contains the codes used to run the experiments presented in our paper. 
In this file, we describe how to obtain the data used for our experiments and the commands used to run experiments with different settings.

Obtaining the data:

	1. For the small-scale experiments, all datasets are available in PyTorch except for USPS for which we obtain the data from https://github.com/mil-tokyo/MCD_DA/tree/master/classification/data.
	2. For large-scale experiments, 
		a. For Imagenet, we follow the instructions present in https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset to download and organize the dataset.
		b. We use CIFAR-10/100 available in PyTorch.
		c. For Aircraft, Pets, and DTD, we follow the instructions of https://github.com/Microsoft/robust-models-transfer#datasets-that-we-use-see-our-paper-for-citations.

Obtaining the pertained models for large-scale experiments:

	1. ResNet18/50 is obtained from PyTorch.
	2. Adversarially trained models are used from https://github.com/Microsoft/robust-models-transfer#download-our-robust-imagenet-models.
	3. CLIP: https://github.com/openai/CLIP
	4. MOCO: https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
	5. SwAV: https://github.com/facebookresearch/swav#model-zoo
	6. SimCLR: https://github.com/google-research/simclr


Running the codes for fine-tuning and task transfer analysis (after navigating to the corresponding folders, run the sample commands). This code corresponds to LearnedAll setting from Sec. 4.2 and illustrates the full working of the algorithm describe in Sec 4.1. Additionally, for the LearnedA setting, set B to be a random permutation matrix and turn off optimization over B and D.  For the FixedAll setting, set B to be a random permutation and turn off optimization over all the variables (A,b, \bar{A}, B, D). (Note: For LearnedA and FixedAll, B and D already satisfy the constraints mentioned in Eq. 3 hence additional softmax is not required and should be removed.)

	A) Commands to run the small-scale experiments:
		1. Training the model on the source dataset MNIST: python3 train_source_model.py --SRC MNIST
		2. Fine-tuning the source model trained on MNIST to the target task of FMNIST: python3 finetune_source_model_on_target_task.py --SRC MNIST --TRG FMNIST --TAU 0.2
		3. Running the task transfer analysis for MNIST as the source and FMNIST as the target task: python3 transfer_source_to_target.py --SRC MNIST --TRG FMNIST --BATCH_SIZE 500 --TAU 0.2

	B) Commands to run the large-scale experiments:
		1. ResNet-18 (vanilla pretrained):
			a. Fine-tune the model on CIFAR-10: python3 finetune_source_model_on_target_task_pytorch_resnet18.py --SRC imagenet_small --TRG cifar10 TAU 0.02
			b. Task transfer analysis: python3 transfer_source_to_target_pytorch_resnet18.py --SRC imagenet_small --TRG cifar10 --TAU 0.02
		
		2. ResNet-18 (adversarially pretrained, eps = 0.1):
			a. Fine-tune the model on CIFAR-10: python3 finetune_source_model_on_target_task_pytorch_resnet18.py --SRC imagenet_small --TRG cifar10 --ROBUST True --EPS 0.1 --TAU 0.02
			b. Task transfer analysis: python3 transfer_source_to_target_pytorch_resnet18.py --SRC imagenet_small --TRG cifar10 --ROBUST True --EPS 0.1 --TAU 0.02

		3. Other Models (simclr, moco, swav, clip):
			a. Fine-tune the model on CIFAR-10: python3 fine-tune.py --SRC imagenet_small TRG cifar10 --model clip
			a. Task transfer analysis: python3 transfer.py --SRC imagenet_small TRG cifar10 --model clip

#### Citing

If you find this useful for your work, please consider citing
<pre>
<code>
@misc{mehra2023analysis,
      title={Analysis of Task Transferability in Large Pre-trained Classifiers}, 
      author={Akshay Mehra and Yunbei Zhang and Jihun Hamm},
      year={2023},
      eprint={2307.00823},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</code>
</pre>
