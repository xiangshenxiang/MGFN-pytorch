# MGFN-pytorch
Here is the code for MGFN model, from the paper Improving Visual Question Answering by Multimodal Gate Fusion Network, which has been accepted for presentation at the International Joint Conference on Neural Networks (IJCNN 2023). The code architecture is created according to the  [OpenVQA](https://github.com/MILVLG/openvqa) platform, where more detailed information can also be found.

## Pretrained models

The performance of the pretrained models on *test-dev* split is reported as follows:

| Overall | Yes/No | Number | Other |
| ------- | ------ | ------ | ----- |
| 71.68   | 87.56  | 56.07  | 61.62 |

The performance of the pretrained models on *test-std* split is reported as follows:

| Overall | Yes/No | Number | Other |
| ------- | ------ | ------ | ----- |
| 72.12   | 87.80  | 55.43  | 62.29 |

These two models can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1CLaIfMSOdQvAMhbqFFxCKkGeASdnS3Oh/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1tdvFHndISw4p1UEvX0TtZA?pwd=8888), and you should unzip and put them to the correct folders as follows:

```
|-- ckpts
	|-- example
	|  |-- epoch12.pkl
```

The evaluations of both the VQA 2.0 *test-dev* and *test-std* splits are run as follows:

```
python3 run.py --RUN='test' --CKPT_V=example --CKPT_E=12
```



## Citation

~~~
@misc{yu2019openvqa,
  author = {Yu, Zhou and Cui, Yuhao and Shao, Zhenwei and Gao, Pengbing and Yu, Jun},
  title = {OpenVQA},
  howpublished = {\url{https://github.com/MILVLG/openvqa}},
  year = {2019}
}
~~~

```
@inProceedings{yu2019mcan,
  author = {Yu, Zhou and Yu, Jun and Cui, Yuhao and Tao, Dacheng and Tian, Qi},
  title = {Deep Modular Co-Attention Networks for Visual Question Answering},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {6281--6290},
  year = {2019}
}
```

```
```

Installation, Data Preparation ,Training, and Evaluation.
Please follow the README of  [MCAN](https://github.com/MILVLG/mcan-vqa) and [OpenVQA](https://github.com/MILVLG/openvqa).
