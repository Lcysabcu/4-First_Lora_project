# 4-Qwen3_Medical_SFT
记录一下第一次使用云服务器跑项目的过程，以及遇到的一些问题。项目源码是[Qwen3微调实战：医疗R1推理风格聊天](https://github.com/Zeyi-Lin/Qwen3-Medical-SFT.git)
## 1.创建容器
在选择的云服务器平台上，点击“创建容器”，选择合适的卡和环境，然后自定义名称。  
创建成功后即为开机状态，可以使用。
## 2.远程连接
打开VScode，安装插件远程资源管理器Remote。  
打开新安装的远程资源管理器，点击SSH一栏的“打开SSH配置文件”，选择第一个默认的config文件，按以下格式填入后保存。相关信息可点击容器的“查看详情”复制。
```bash
Host 自定义名称
    HostName IP地址
    User 用户名
    Port 端口号
```
保存刷新后，可在SSH一栏下看见刚刚新建的远程，点击“在当前窗口中连接”，选择第一个Linux，根据要求输入密码，等待一段时间即可连接成功。  
可在终端输入以下指令查看环境中已安装的包。
```bash
conda list
```
## 3.上传项目文件
打开已安装的FileZilla，点击左上角“文件”下方的“打开站点管理器”，点击“新站点”，在右方复制粘贴容器的IP地址、端口、用户名，协议选择第二个SFTP，点击“连接”后按要求输入密码，即可连接成功。  
在左边窗口中找到项目文件夹，将其拖到右边窗口的root文件夹，等待所有文件上传成功。
## 4.运行项目
在VScode中，点击“打开项目文件夹”，选择刚刚上传的项目文件夹。
### 4.1 环境搭建
在终端输入以下指令新建一个虚拟环境。
```bash
conda create --name 自定义环境名 python=所需的版本
```
创建虚拟环境后，在终端输入以下指令进入环境。
```bash
conda activate  环境名
```
在终端输入以下指令安装项目所需的包。
```bash
pip install -r requirements.txt
```
如果遇到问题，则把报错复制给Deepseek，按回复解决即可。比如遇到网络问题，则可使用国内镜像源安装，指令如下。
```bash
# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```
### 4.2 数据准备
在终端输入以下指令以准备数据。
```bash
python data.py
```
### 4.3 训练
在终端输入以下指令，Lora微调训练模型。
```bash
python train_lora.py
```
项目源码中采用Swanlab可视化训练过程，但在运行过程中无法复制自己的Swanlab API至终端，故选择[3]跳过即可继续训练。
### 4.4 推理
在终端输入以下指令，查看模型训练效果。
```bash
python inference_lora.py
```
## 5.下载项目文件
打开FileZilla，点击左上角“文件”下方的“打开站点管理器”，选择之前建立的站点，点击“连接”后按要求输入密码，即可连接成功。  
在右边窗口的root文件夹下找到项目文件夹，将其拖到左边窗口，等待所有文件下载成功。
## 6.关闭容器
所有步骤完成后，记得将容器关机。
