# Fast-BEV: A Fast and Strong Bird’s-Eye View Perception Baseline

A PaddlePaddle Implementation for Fast-BEV 


## <h2 id="4">训练 & 评估</h2>

### <h3 id="41">nuScenes数据集</h3>

#### 数据处理

- 目前Paddle3D中提供的Fast-BEV模型支持在nuScenes数据集上训练，因此需要先准备nuScenes数据集，请在[官网](https://www.nuscenes.org/nuscenes)进行下载，并且需要下载CAN bus expansion和MAP expansion数据，将数据集目录准备如下：

```
nuscenes_dataset_root
|—— can_bus
|—— samples  
|—— sweeps  
|—— maps  
|—— v1.0-trainval  
```

在Paddle3D的目录下创建软链接 `datasets/nuscenes`，指向到上面的数据集目录:

```
mkdir datasets
ln -s /path/to/nuscenes_dataset_root ./datasets
mv ./datasets/nuscenes_dataset_root ./datasets/nuscenes
```

为加速Fast-BEV训练过程中Nuscenes数据集的加载和解析，需要事先将Nuscenes数据集里的标注信息并时间融合转换成4D索引存储在`pkl`后缀文件中。执行以下命令会生成`nuscenes_infos_train_4d_interval3_max60.pkl`和` nuscenes_infos_train_4d_interval3_max60.pkl`：

```
python tools/create_fastbev_nus_infos_seq_converter.py --dataset_root ./datasets/nuscenes --can_bus_root ./datasets/nuscenes --save_dir ./datasets/nuscenes \
--mode train --max_adj 60 --interval 3
```

**注** ：若为mini数据集则`--mode`参数改为`mini_train`