# demo2.py

from mmcls.apis import inference_model, init_model, show_result_pyplot

# 指定配置文件
config_file = 'configs/resnet/resnet50_8xb32_in1k.py'
# 指定checkpoint文件
checkpoint_file = 'checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

# 指定GPU
device = 'cuda:0'
# 根据配置文件和模型文件搭建模型
model = init_model(config_file, checkpoint_file, device=device)
# 测试一张图片，要确保这个路径下有这张图
img = 'demo/bird.jpeg'
result = inference_model(model,img)
# 展示结果，如果是远程ssh的话，下面这句注释掉，因为显示不了
show_result_pyplot(model,img,result)
