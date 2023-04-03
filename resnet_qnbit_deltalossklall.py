#This file calculates deltaloss.
import torch
import csv
import imagenet
import resnet
import functions
import time

#Weight bitwidth For example, 8 shows 8-bit bitwidth.You can be specified an integer value. 
w_bit = 8
#speed-up training and inference
torch.backends.cudnn.benchmark=True
#Use cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#define net.
net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
#net = resnet.resnet34(num_classes=1000, pretrained='imagenet')
layers = [net.layer1,net.layer2,net.layer3,net.layer4]
#Calculate original model deltaloss
netloss = functions.evaluate_loss(net, device, imagenet.val_loader)
print('preloss=',netloss)

count = 1 #layernum
#List for csv file outputs
printlayernum=[] #layernum
printchannelnum=[] #channelnum
printdeltalosses=[] #deltaloss

start = time.time() #Time measurement

for layer_index in range(len(layers)):
    for block_index in range(len(layers[layer_index])):
        for i in range(layers[layer_index][block_index].conv1.out_channels): #odd conv layer
            #net initialized
            net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
            #net = resnet.resnet34(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            #quantization
            layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, i)
            #evaluate
            loss = functions.evaluate_loss(net, device, imagenet.val_loader)
            deltaloss = loss - netloss
            print('layernum=',count,'channelnum=',i+1,'deltaloss=',deltaloss)
            #lists append.
            printlayernum.append(count)
            printchannelnum.append(i+1)
            printdeltalosses.append(deltaloss)
        count += 1

        for j in range(layers[layer_index][block_index].conv2.out_channels): #even conv layer
            #net initialized
            net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
            #net = resnet.resnet34(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            #quantization
            layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, j)
            #evaluate
            loss = functions.evaluate_loss(net, device, imagenet.val_loader)
            deltaloss = loss - netloss
            print('layernum=',count,'channelnum=',j+1,'deltaloss=',deltaloss)
            #lists append.
            printlayernum.append(count)
            printchannelnum.append(j+1)
            printdeltalosses.append(deltaloss)
        count += 1

end = time.time() #Time measurement
elapsed_time = end - start
print('elapsed_time:', elapsed_time, 'sec')
print('done')

#csv file output
with open("resnet18_deltaloss.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(printlayernum)
    writer.writerow(printchannelnum) 
    writer.writerow(printdeltalosses)
