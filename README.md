<p align="center">
  <img src="asset/logo.jpg"  height=200>
</p>

### <div align="center"> 🚀 Dimba: Transformer-Mamba Diffusion Models <div> 

<div align="center">
  <a href="https://github.com/feizc/Dimba/"><img src="https://img.shields.io/static/v1?label=Dimba Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://dimba-project.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=red&logo=github-pages"></a> &ensp;
    <a href="https://huggingface.co/feizhengcong/Dimba"><img src="https://img.shields.io/static/v1?label=models&message=HF&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/feizhengcong/Dimba"><img src="https://img.shields.io/static/v1?label=dataset&message=HF&color=green"></a> &ensp;
    <a href="http://arxiv.org/abs/2406.01159"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Dimba&color=purple&logo=arxiv"></a> &ensp;
</div>

---

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper Transformer-Mamba Diffusion Models. You can find more visualizations on our project page.

<b> TL; DR: Dimba is a new text-to-image diffusion model that employs a hybrid architecture combining Transformer and Mamba elements, thus capitalizing on the advantages of both architectural paradigms.</b>

---


![some generated cases.](asset/case.jpg)



---


## 1. Environments

- Python 3.10
  - `conda create -n your_env_name python=3.10`

- Requirements file
  - `pip install -r requirements.txt`

- Install ``causal_conv1d`` and ``mamba``
  - `pip install -e causal_conv1d`
  - `pip install -e mamba`

## 2. Download Models

Models reported in paper can be directly dounloaded as follows. 
| Model                       | #Params | url      | 
|:----------------------------|:--------|:----------------------------------------------------------------------------------------------------------------|
| t5                          | 4.3B     |[huggingface](https://huggingface.co/)|
| vae                          | 80M     |[huggingface](https://huggingface.co/)|
| Dimba-L-512                  | 0.9B     |[huggingface](https://huggingface.co/)|
| Dimba-L-1024                  | 0.9B     |[huggingface](https://huggingface.co/)|
| Dimba-L-2048                  | 0.9B     |[huggingface](https://huggingface.co/)|
| Dimba-G-512                  | 1.8B     |[huggingface](https://huggingface.co/)|
| Dimba-G-1024                  | 1.8B     |[huggingface](https://huggingface.co/)|

## 3. Inference

## 4. Training 


## 5. Acknowledgments

The codebase is based on the awesome [PixArt](https://github.com/PixArt-alpha/PixArt-alpha), [Vim](https://github.com/hustvl/Vim), and [DiS](https://github.com/feizc/DiS) repos. 


