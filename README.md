# Deformable-Detr.axera
Deformable-Detr DEMO on Axera NPU.

### 1. 工程下载
```  
git clone https://github.com/AXERA-TECH/deformable-detr.axera.git
```

### 2. 模型转换
```
pulsar2 build --config  ./config/config.json
```
### 3. 板端运行
```
python inference.py --model ./output/detr.axmodel --img ./assets/bus.jpg --output out.jpg --thresh 0.6
```
### 4. 结果展示
![result](./output/out.jpg)
