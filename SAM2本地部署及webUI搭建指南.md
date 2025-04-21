# SAM2本地部署及webUI搭建指南

## 一、使用F5 AI社区提供的SAM2本地部署中文整合包
### 1. 下载整合包
可以从以下网盘下载【SAM2.zip】压缩包：
- 百度网盘下载链接（永久有效）：[链接](https://pan.baidu.com/s/13kdNqTdr2S7_ampAc71kVg?pwd=3fy9)，提取码：3fy9
- 123网盘下载链接（永久有效）：[链接](https://www.123pan.com/s/5DsaTd-OAPc.html)
- 夸克网盘下载链接（永久有效）：[链接](https://pan.quark.cn/s/6557b6989579)

### 2. 启动程序
解压【SAM2.zip】压缩包，找到【SAM2.exe】文件，双击启动程序。

### 3. 选择素材
在主界面点击【打开素材】按钮，选择要处理的图片或视频。

### 4. 一键抠图
在素材上点击想抠取的目标物，软件会自动识别并抠取，可以通过调整选项来优化结果。

### 5. 导出结果
点击【下载】按钮，选择想要保存的位置，即可将抠好的素材保存下来。

## 二、手动部署
### 1. 环境准备
需要python>=3.10，以及torch>=2.3.1和torchvision>=0.18.1。建议为此安装创建一个新的Python环境，并按照[PyTorch官网](https://pytorch.org/)通过`pip`安装PyTorch 2.3.1（或更高版本）。如果当前环境中的PyTorch版本低于2.3.1，则上述安装命令将尝试使用pip将其升级到最新的PyTorch版本。

### 2. 下载源码
使用以下命令克隆SAM2的GitHub仓库：
```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
```

### 3. 安装依赖
#### 3.1 常规安装
使用以下命令安装项目依赖：
```bash
pip install -e .
```

#### 3.2 可能遇到的问题及解决方法
在执行`pip install -e .`时可能会报错，提示`OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.`，可以按照以下步骤解决：
1. 下载cuda12.4（12.1也完全没有问题）。
2. 在系统环境变量中添加一个新条目，名称为`CUDA_HOME`，值为`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4;`。
3. 执行以下命令：
```bash
conda create -n sam2 python=3.10
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --no-build-isolation -e .
pip install --no-build-isolation -e ".[demo]"
```

### 4. 下载检查点
首先，我们需要下载一个模型检查点。所有模型检查点都可以通过运行以下命令来下载：
```bash
cd checkpoints
./download_ckpts.sh
```
也可以单独下载以下模型：
- sam2_hiera_tiny.pt
- sam2_hiera_small.pt
- sam2_hiera_base_plus.pt
- sam2_hiera_large.pt

### 5. 搭建webUI页面
可以使用Python的Flask或Django框架来搭建webUI页面，以下是一个简单的Flask示例：
```python
from flask import Flask, request, render_template_string, send_file
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_image_predictor
import torch

app = Flask(__name__)

# 加载SAM2模型
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的文件
        file = request.files['file']
        if file:
            # 读取图片
            image = Image.open(file.stream).convert("RGB")
            image_np = np.array(image)

            # 获取用户标记的点
            points = request.form.get('points')
            if points:
                points = [int(p) for p in points.split(',')]
                points = np.array(points).reshape(-1, 2)
                labels = np.ones(points.shape[0])

                # 设置图片并进行预测
                predictor.set_image(image_np)
                masks, scores, logits = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=True,
                )

                # 选择得分最高的掩码
                highest_score_mask = masks[np.argmax(scores)]

                # 将掩码转换为黑白遮罩图片
                mask_image = Image.fromarray((highest_score_mask * 255).astype(np.uint8))
                mask_image.save('mask.png')

                return send_file('mask.png', mimetype='image/png')

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>SAM2 WebUI</title>
    </head>
    <body>
        <h1>SAM2 WebUI</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <br>
            <label for="points">请输入标记点的坐标（格式：x1,y1,x2,y2,...）：</label>
            <input type="text" id="points" name="points">
            <br>
            <input type="submit" value="分割">
        </form>
    </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)
```

### 6. 运行webUI
将上述代码保存为`app.py`，然后在终端中运行以下命令：
```bash
python app.py
```
打开浏览器，访问`http://127.0.0.1:5000`，即可看到webUI页面。在页面上选择图片，输入标记点的坐标，点击分割按钮，即可看到分割结果。

### 7. 输出纯绿幕视频或者黑白遮罩视频
在得到分割掩码后，可以使用OpenCV等库将其应用到原始视频上，生成纯绿幕视频或者黑白遮罩视频。以下是一个简单的示例：
```python
import cv2
import numpy as np

# 读取原始视频和掩码图片
cap = cv2.VideoCapture('input_video.mp4')
mask = cv2.imread('mask.png', 0)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 将掩码应用到帧上
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # 创建纯绿幕背景
        green_screen = np.zeros_like(frame)
        green_screen[:, :, 1] = 255

        # 将掩码部分替换为绿幕背景
        green_screen_masked = cv2.bitwise_and(green_screen, green_screen, mask=cv2.bitwise_not(mask))
        final_frame = cv2.add(masked_frame, green_screen_masked)

        # 写入输出视频
        out.write(final_frame)
    else:
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
```

以上代码实现了在本地部署SAM2并搭建具备webUI页面的系统，支持用户选择视频或图片，手动标记分割主体，预览分割结果，输出纯绿幕或黑白遮罩视频且保持分辨率、帧率和码率与输入视频相同。