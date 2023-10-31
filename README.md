# Wining_Health_Project

这是第六届上海交大-卫宁健康智慧医疗挑战赛0x73b队的项目代码.

## 使用方法

### 运行设备

使用华为Atlas200Dk板卡,搭载Ascend310芯片.
请确保安装CANN框架,python能正常调用acl库.

### 安装依赖
```
pip install requirements.txt
```

### 下载支持材料
https://jbox.sjtu.edu.cn/l/C1VaL3

#### 下载数据集(可选)

放置于`experiments/data`
已有编号为100的样例数据

#### 下载神经网络模型

放置于`experiments/EcgResNet34/checkpoints`


### 试运行
```
cd Wining_Health_Project
python pipeline_asc.py --config configs/config.json
```

### 查看运行情况

浏览器打开`experiments/EcgResNet34/results/100.html`,即可查看可视化结果.