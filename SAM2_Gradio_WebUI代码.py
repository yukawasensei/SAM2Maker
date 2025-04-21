# 导入必要的库
import gradio as gr
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

# 加载SAM2模型
sam2_checkpoint = "checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# 定义处理视频帧的函数
# 定义处理视频帧的函数
# 定义处理视频帧的函数
def process_frame(frame, points):
    image_np = np.array(frame)
    if points:
        points = np.array(points).reshape(-1, 2)
        labels = np.ones(points.shape[0])
        predictor.set_image(image_np)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        highest_score_mask = masks[np.argmax(scores)]
        # 生成黑白遮罩视频帧
        mask_frame = (highest_score_mask * 255).astype(np.uint8)
        # 生成纯绿幕视频帧
        green_screen = np.zeros_like(image_np)
        green_screen[:, :, 1] = 255
        green_screen_masked = cv2.bitwise_and(green_screen, green_screen, mask=cv2.bitwise_not(mask_frame))
        masked_frame = cv2.bitwise_and(image_np, image_np, mask=mask_frame)
        green_screen_frame = cv2.add(masked_frame, green_screen_masked)
        return mask_frame, green_screen_frame
    return None, None

# 定义Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# SAM2视频分割WebUI")
    with gr.Row():
        video_input = gr.Video(label="输入视频")
        point_input = gr.Textbox(label="标记点坐标（格式：x1,y1,x2,y2...）")
    with gr.Row():
        mask_output = gr.Image(label="黑白遮罩视频帧")
        green_screen_output = gr.Image(label="纯绿幕视频帧")
    process_button = gr.Button("处理视频帧")

    process_button.click(
        fn=lambda video, points: process_frame(cv2.cvtColor(cv2.imread(video), cv2.COLOR_BGR2RGB), [int(p) for p in points.split(',')]) if points else (None, None),
        inputs=[video_input, point_input],
        outputs=[mask_output, green_screen_output]
    )

if __name__ == "__main__":
    demo.launch()
