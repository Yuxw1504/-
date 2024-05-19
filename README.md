# 表情包推荐系统
基于transformer实现的表情包推荐系统

项目目标：实现类似于QQ微信中，输入一段文字，就会检索你这段文字所需要的表情包的项目

项目数据来源:https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp

目前实现的项目:

通过输入的文本(目前仅支持英文文本),识别表情标签，然后推荐相关图片(没有表情包库)


项目操作步骤：

1.下载数据

2.对数据进行预处理

3.训练模型

4.运行app.py

5.访问127.0.0.1:5001进行使用


未来优化

1.实现表情包的推荐

2.实现实时文本表情包推荐


实际效果展示
![WeChat5db311b3352e85ee103416394b237d61](https://github.com/Yuxw1504/Emotion-Recognition/assets/163949558/e863cb27-e39b-4cbe-b755-5cbf5536ee0c)
在输入栏输入 Are you ok?
得到感情标签以及推荐图片（表情包）
![WeChatbd41d04331fdca04ca2c860630aa316b](https://github.com/Yuxw1504/Emotion-Recognition/assets/163949558/a0a8cfc7-1cd7-466e-83aa-fb363565c9c2)
