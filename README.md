# simple-lama-inpainting

<div align="center">
Simple pip package for LaMa[1] inpainting.<br>
<a href="https://badge.fury.io/py/simple-lama-inpainting"><img src="https://badge.fury.io/py/simple-lama-inpainting.svg" alt="PyPI version" height="18"></a>
</div>

## Installation
```
pip install simple-lama-inpainting
```

## Usage
### 
```
运行remove_watermark_image.py，这个调用后会选择test中的测试图片做去水印
```
### 效果
```
原图：
![效果对比](output_inpainted_image.jpg)
去水印后图片：
![效果对比](tests/data_self/test_1.jpg)
```


## Sources
[1] Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K., & Lempitsky, V. (2021). Resolution-robust Large Mask Inpainting with Fourier Convolutions. arXiv preprint arXiv:2109.07161. \
[2] https://github.com/saic-mdal/lama \
[3] https://github.com/Sanster/lama-cleaner
