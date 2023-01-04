# [YOLOv4说明](#目录)

YOLOv4作为先进的检测器，它比所有可用的替代检测器更快（FPS）并且更准确（MS COCO AP50 ... 95和AP50）。
本文已经验证了大量的特征，并选择使用这些特征来提高分类和检测的精度。
这些特性可以作为未来研究和开发的最佳实践。

[论文](https://arxiv.org/pdf/2004.10934.pdf)：
Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal Speed and Accuracy of Object Detection[J]. arXiv preprint arXiv:2004.10934, 2020.

# [模型架构](#目录)

选择CSPDarknet53主干、SPP附加模块、PANet路径聚合网络和YOLOv4（基于锚点）头作为YOLOv4架构。


# [数据集](#目录)



- 目录结构如下，由用户定义目录和文件的名称：

    ```text
        ├── dataset
            ├─train
            │   ├─picture1.jpg
            │   ├─picture1.xml
            │   ├─ ...
            │   ├─picturen.jpg
            │   └─picturen.xml       
            ├─test
            │   ├─picture1.jpg
            │   ├─picture1.xml
            │   ├─ ...
            │   ├─picturen.jpg
            │   └─picturen.xml  
    ```


# [评估过程]

## 验证

```bash
python eval_xml.py  --xml_dir  ../dataset/test  \     
                  --jpg_src_path ../dataset/test \    # 
                  --predict_result ./predict_result \ 
                  --pretrained  ./best_map.ckpt        
                  
```
- predict_result：输出推理xml文件
- pretrained：推理模型
- xml_dir：输入xml路径
- jpg_src_path：输入图片路径

推理结果保存在脚本执行的当前路径，可以在控制台中看到精度计算结果。

```text
=============coco eval reulst=========
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.646
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.919
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.788
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.549
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.679
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.304
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.640
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.698
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.624
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.724
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
```
