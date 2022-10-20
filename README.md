# Robust Camera Trace

An official implementation code for paper "Robust Camera Model Identification over Online Social Network Shared Images via Multi-Scenario Learning"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)


## Background
Camera model identification (CMI) can be widely used in image forensics such as authenticity determination, copyright protection, forgery detection, etc. Meanwhile, with the vigorous development of the Internet, online social networks (OSNs) have become the dominant channels for image sharing and transmission. However, the inevitable lossy operations on OSNs, such as compression and post-processing, impose great challenges to the existing CMI schemes, as they severely destroy the camera traces left in the images under investigation. In this work, we propose a novel CMI method that is robust against the lossy operations of various OSN platforms. It is observed that the camera trace extractor can be easily trained on a single degradation scenario (e.g., one specific OSN platform); while much more difficult on mixed degradation scenarios (e.g., multiple OSN platforms). Inspired by this observation, we design a new multi-scenario learning (MSL) strategy, enabling us to extract robust camera traces across different OSNs. Furthermore, noticing that image smooth regions incur less distortions by OSN and less interference by image signal itself, we suggest a SmooThness-Aware Trace Extractor (STATE) that can adaptively extract camera traces according to the smoothness of the input image.

<p align='center'>  
  <img src='https://github.com/HighwayWu/CameraTraceOSN/blob/master/imgs/framework.jpg' width='870'/>
</p>
<p align='center'>  
  <em>Framework of the proposed method.</em>
</p>


The superiority of our method is verified by comparative experiments with four state-of-the-art methods [Kuzin (ICBD'18)](https://ieeexplore.ieee.org/iel7/8610059/8621858/08622031.pdf), [NoiPri (TIFS'19)](https://ieeexplore.ieee.org/iel7/10206/4358835/08713484.pdf), [ForSim (TIFS'19)](https://ieeexplore.ieee.org/iel7/10206/4358835/08744262.pdf), and [PCN (SPL'20)](https://ieeexplore.ieee.org/iel7/97/4358004/09141509.pdf), especially under various OSN transmission scenarios. Particularly, for the open-set camera model verification task, we greatly surpass the second-place by 15.30\% AUC on the [**FODB**](https://link.springer.com/chapter/10.1007/978-3-030-68780-9_40) dataset; while for the close-set camera model classification task, we are significantly ahead of the second-place with 34.51\% F1 score on the [**SIHDR**](https://www.mdpi.com/361986) dataset.

<p align='center'>
  <img src='https://github.com/HighwayWu/CameraTraceOSN/blob/master/imgs/cmp.png' width='500'/>
</p>
<p align='center'>
  <em>Performance on open-set verification task of our proposed method, compared with four state-of-the-art schemes. It should be noted that both the test dataset FODB and the considered OSN platforms (Twitter, Wechat, QQ, Telegram, and Dingding) are *unknown* at training time, mimicking a practical situation.</em>
</p>

Besides, we build new OSN-transmitted datasets over 9 popular OSNs (Twitter, Telegram, Whatsapp, Instagram, Facebook, Weibo, QQ, Dingding, and Wechat) based on the existing camera datasets [**FODB**](https://link.springer.com/chapter/10.1007/978-3-030-68780-9_40) and [**SIHDR**](https://www.mdpi.com/361986), for not only evaluating the robustness of existing CMI algorithms, but also beneficial different applications in the forensic community. The OSN-transmitted datasets can be downloaded from: [**FODB** (Google Drive)](https://drive.google.com/file/d/1rWbcNa4_EbY-O_nhkaCmKKT5rci5lk5p/view?usp=sharing) or [**FODB** (Baidu Pan)](), [**SIHDR** (Google Drive)](https://drive.google.com/file/d/1m8VOCAsUsPu5SKb5kK9bWlfV_-VYCsq8/view?usp=sharing) or [**SIHDR** (Baidu Pan)]().

## Dependency
- torch 1.11.0

## Usage

1) Preparation:
Download the train/test dataset and put in the `data/`. Then run:
```bash
python preprocess.py
```

2) Train:
```bash
python train.py
```
**Note: According to the project requirements, the training code will be released later.**

3) To test the model for Trace Extraction:
```bash
python test_TraceExtraction.py
```

4) To test the model for Open-set Verification:
```bash
python test_OpenSetVerification.py
```


## Citation

If you use this code/dataset for your research, please consider citing the references of the original dataset:
```
@inproceedings{fodb2021,
  title={The Forchheim image database for camera identification in the wild},
  author={B. Hadwiger and C. Riess},
  booktitle={Proc. Int. Conf. Pattern Recognit.},
  pages={500--515},
  year={2021},
  organization={Springer}
}

@article{sihdr2018,
  title={A new dataset for source identification of high dynamic range images},
  author={O. Shaya and P. Yang and R. Ni and Y. Zhao and A. Piva},
  journal={Sensors},
  volume={18},
  number={11},
  pages={3801},
  year={2018},
  publisher={MDPI}
}
```


## Acknowledgments
- Part of the codes are based on the [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo) and the [LibMTL](https://github.com/median-research-group/LibMTL).
- Alibaba Innovative Research Program
