# HyperMono: A Monotonicity-aware Approach to Hyper-Relational Knowledge Representation
#### This repo provides the source code & data of our paper: HyperMono: A Monotonicity-aware Approach to Hyper-Relational Knowledge Representation.
## Dependencies
* conda create -n hypermono python=3.7 -y
* PyTorch 1.8.1
* contiguous_params 1.0.0
* scipy 1.7.3
* tqdm 4.64.1
* fastmoe 0.2.0
  * download the [fastmoe](https://github.com/laekov/fastmoe) project
  * cd fastmoe folder
  * conda install "gxx_linux-64<=10" nccl -c conda-forge -y 
  * pip install -e .
  * If you have problems using MoE, you can directly download the one we used [fastmoe](https://drive.google.com/file/d/1c3ijOe5PacVWyfmD2amUk0dpTbsjIHIS/view?usp=sharing).
* **Note:** We need to emphasize that you need to use the same experimental environment as us, especially the same version of Pytorch. We conducted the experiment on version **Pytorch 1.8.1**. In addition, we also test our code on **Pytorch 1.11.0**, and the experimental results are fine too. However, on **Pytorch 2.1.2**, the MRR indicator value will be equal to 0, which is completely unreasonable, so we recommend that you use the same environment as us.
### Before Training model
* You need to unzip the compressed file under the dataset folder;
### Training model
Taking the WikiPeople dataset as an example, you can run the following script：
```python
sh run.sh
```
### Training logs
* **Note:** We provide logs of our training in the logs directory.
