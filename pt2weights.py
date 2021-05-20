from models import *
# //注意找到工程文件中有没有 models.py文件 例如下面Darknet save_weights函数都在models.py文件里定义好了  直接调用就好  注意：新建的pt2weights.py与models.py在同一目录下
model=Darknet("cfg/yolov3.cfg")
# //训练时候用的cfg 改成自己.cfg的地址
# load_darknet_weights(model,"weights/best.pt")
# save_weights(model,path='weights/latest.weights',cutoff=-1)
checkpoint = torch.load("weights/last.pt",map_location='cpu')
model.load_state_dict(checkpoint['model'])
save_weights(model,path='weights/last.weights',cutoff=-1)
