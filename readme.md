<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Implementation CV2 Haarcascade Object Detection Models 
 
## __Table Of Content__
- [__Brief__](#brief)
  - [__Project__](#project)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-cv2-haarcascade-objectdetection.hf.space)
  - [__Study__](#study) -> [Colab](https://colab.research.google.com/drive/1HG0-eigOKrC1ViUaE_vvlMplEBMO9OQV)
    - [__Dependencies__](#a-dependencies)
    - [__Models__](#model-list)
    - [__File Structure__](#file-structures)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is an __Object Detection__ project that uses opencv Haarcascade pretrained models (whole haarcascades) to __detect related objects__ .
- The __goal__ is that implementing all the opencv pretrained object detection models for __detecting related objects__. 

#### __Overview__
- This project involves getting all the opencv pretrained haarcascade object detection deep learning models then using on any data according to user demand. It just an implementation of the pretrained models  The project uses Python and several popular libraries such as Pandas, NumPy, OpenCV.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-cv2-haarcascade-objectdetection.hf.space" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1Pw7c2DEwIFS1u34eNLpjgSqpOfTa0_lD#scrollTo=u9VBBQQWKweX"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/CV2_Haarcascade_ObjectDetection/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1HG0-eigOKrC1ViUaE_vvlMplEBMO9OQV"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __Detecting objects__ which related corresponding models.
    - __Usage__: upload the image and select the model from option dropdawn button to detect the objects.
- Embedded [Demo](https://ertugruldemir-cv2-haarcascade-objectdetection.hf.space) window from HuggingFace Space
    

<iframe
	src="https://ertugruldemir-cv2-haarcascade-objectdetection.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Dependencies__:
  - The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from tensoflow.

#### Model List
- Pretrained Object Detection Haarcascade models from [opencv github repo](https://github.com/opencv/opencv/tree/master/data/haarcascades) .
  - (1) "eye"
  - (2) "eye_tree_eyeglasses"
  - (3) "frontalcatface"
  - (4) "frontalcatface_extended"
  - (5) "frontalface_alt"
  - (6) "frontalface_alt2"
  - (7) "frontalface_alt_tree"
  - (8) "frontalface_default"
  - (9) "fullbody"
  - (10) "lefteye_2splits"
  - (11) "lowerbody"
  - (12) "profileface"
  - (13) "righteye_2splits"
  - (14) "smile"
  - (15) "upperbody"


### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── examples
│   │   ├── messi.jpeg
│   │   └── ronaldo.jpeg
│   └── requirements.txt
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── readme.md
└── study.ipynb

```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/examples
    - Example cases to test the model.
  - demo_app/requirements.txt
    - It includes the dependencies of the demo_app.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.     



## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

