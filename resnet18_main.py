import torch
import csv
import imagenet
import functions
import time
import resnet
import os

from get_commandlines import get_args
args = get_args()
#speed-up training and inference
torch.backends.cudnn.benchmark=True

# weight bit-width
w8_bit=8
w6_bit=6
w4_bit=4
w2_bit=2
w1_bit=1
w0_bit=0

#parameters initialized
beta = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#list initialized
valuationds=[]
valuationks=[]
valuationlayers=[]
valuationnexts=[]
valuationfirsts=[]
valuationpostponings=[]
printparams=[]
printaccs=[]
printkls=[]
printw_bits=[]
printinferenceflags=[]
printlayernum=[]
printchannels=[]

pthname='resnet18_semilayerdlossplusminus_postponing.pth'

start = time.time()

#model initialized
net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
net2 = resnet.resnet18(num_classes=1000, pretrained='imagenet')
layers = [net.layer1,net.layer2,net.layer3,net.layer4]
layer2s = [net2.layer1,net2.layer2,net2.layer3,net2.layer4]
preacc,netloss,originaloutputs = functions.evaluate_acc_loss_softmax(net2, device, imagenet.val_loader)
printparams.append(0)
printaccs.append(preacc)
printkls.append(0.0)
printw_bits.append(0)
printinferenceflags.append(0)
printlayernum.append(0)
printchannels.append(0)
print('preacc=',preacc)

#make list about deltaloss
count=0
with open("dataset/ResNet18ImageNetq86421bitprudeltaloss.csv",encoding = "utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if count == 0: #layernum
            first = row
        elif count == 1:#channelnum
            second = row
        elif count == 2:
            third = row
        elif count == 3:
            four = row
        elif count == 4:
            five = row
        elif count == 5:
            six = row
        elif count == 6:
            seven = row
        else:
            eight = row
        count += 1
    
    for num in range(len(first)):
        lnum = int(first[num])
        cnum = int(second[num])-1
        dl8value = float(third[num])
        dl6value = float(four[num])
        dl4value = float(five[num])
        dl2value = float(six[num])
        dl1value = float(seven[num])
        dl0value = float(eight[num]) #pruning

        if lnum==1 or lnum==2:
            layer_index=0
            block_index=0
        elif lnum==3 or lnum==4:
            layer_index=0
            block_index=1
        elif lnum==5 or lnum==6:
            layer_index=1
            block_index=0
        elif lnum==7 or lnum==8:
            layer_index=1
            block_index=1
        elif lnum==9 or lnum==10:
            layer_index=2
            block_index=0
        elif lnum==11 or lnum==12:
            layer_index=2
            block_index=1
        elif lnum==13 or lnum==14:
            layer_index=3
            block_index=0
        else:
            layer_index=3
            block_index=1

        #normalized deltaloss part
        if lnum%2!=0:
            param=layers[layer_index][block_index].conv1.weight[cnum].data.numel()
        else:
            param=layers[layer_index][block_index].conv2.weight[cnum].data.numel()
        if dl8value<0:
            dl8value*=(param*0.75)
        else:
            dl8value/=(param*0.75)
        if dl6value<0:
            dl6value*=(param*0.8125)
        else:
            dl6value/=(param*0.8125)
        if dl4value<0:
            dl4value*=(param*0.875)
        else:
           dl4value/=(param*0.875)
        if dl2value<0:
            dl2value*=(param*0.9375)
        else:
            dl2value/=(param*0.9375)
        if dl1value<0:
            dl1value*=(param*0.96875)
        else:
            dl1value/=(param*0.96875)
        if dl0value<0:
            dl0value*=param
        else:
            dl0value/=param
        valuationds.append([layer_index,block_index,lnum,cnum,dl8value,dl6value,dl4value,dl2value,dl1value,dl0value])
print('deltaloss list done.')

#make list about KL divergence
count=0
with open("dataset/ResNet18ImageNetq86421bitprukl.csv",encoding = "utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        if count == 0: #layernum
            first = row
        elif count == 1:#channelnum
            second = row
        elif count == 2:
            third = row
        elif count == 3:
            four = row
        elif count == 4:
            five = row
        elif count == 5:
            six = row
        elif count == 6:
            seven = row
        else:
            eight = row
        count += 1
    
    for num in range(len(first)):
        lnum = int(first[num])
        cnum = int(second[num])-1
        kl8value = float(third[num])
        kl6value = float(four[num])
        kl4value = float(five[num])
        kl2value = float(six[num])
        kl1value = float(seven[num])
        kl0value = float(eight[num]) #pruning

        if lnum==1 or lnum==2:
            layer_index=0
            block_index=0
        elif lnum==3 or lnum==4:
            layer_index=0
            block_index=1
        elif lnum==5 or lnum==6:
            layer_index=1
            block_index=0
        elif lnum==7 or lnum==8:
            layer_index=1
            block_index=1
        elif lnum==9 or lnum==10:
            layer_index=2
            block_index=0
        elif lnum==11 or lnum==12:
            layer_index=2
            block_index=1
        elif lnum==13 or lnum==14:
            layer_index=3
            block_index=0
        else:
            layer_index=3
            block_index=1

        valuationks.append([layer_index,block_index,lnum,cnum,kl8value,kl6value,kl4value,kl2value,kl1value,kl0value])
        valuationlayers.append([layer_index,block_index,lnum,cnum,w8_bit,0,32,num+1]) #0:layer_index,1:block_index,2:layernum,3:channelnum,4:quantized bit,5:flag,6:selected bit,7:all channels index
print('KL divergence list done.')


#deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,4) #memo each 2-D matrix
valuationlayers=[]
#classified into two groups. deltaloss plus or minus and 0.
#3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
#use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)        
#list sort
valuationfirsts=functions.make_quantizedlists(semilayers,orders)


#parameter initialized
params=0
param=0
lists=[]
flag=0
inferenceflag=0
counta=0
postponingflag=1
#mix-precision part
print('8bit inference start')
for i in range(len(valuationfirsts)):
    layer_index=valuationfirsts[i][0]
    block_index=valuationfirsts[i][1]
    lnum=valuationfirsts[i][2]
    cnum=valuationfirsts[i][3]
    w_bit=valuationfirsts[i][4]
    signflag=valuationfirsts[i][5]
    selectedbit=valuationfirsts[i][6]
    index=valuationfirsts[i][7]

    if lnum==100:
        break
    if lnum%2!=0 and w_bit!=0:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    if lnum%2==0 and w_bit!=0:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    #append part
    lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
    counta+=1

    if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]:
        acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
        kldiv = functions.KLdiv(originaloutputs,afteroutputs)
        print('preacc=',preacc)
        print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
        if acc>=preacc:
            flag=1
            postponingflag=0
            torch.save(net.state_dict(),pthname)
            preacc=acc
            params+=param
            printparams.append(params)
            printaccs.append(acc)
            printkls.append(kldiv)
            printw_bits.append(w_bit)
            printinferenceflags.append(inferenceflag)
            printlayernum.append(lnum)
            printchannels.append(counta)              
            print('success: accuracy=',acc,'kl divergence=',kldiv,'debug param=',param)
            for value in range(len(lists)):
                valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w6_bit,lists[value][5],w_bit,lists[value][7]]) 
        else: #acc<preacc->postponing
            print('postponing/reduced number of param:',param)
            if flag==0:
                net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
            else:
                weight = torch.load(pthname)
                net.load_state_dict(weight)
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            debugacc,debugloss,debugoutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            print('debug=',debugacc)
            for value in range(len(lists)):
                valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w6_bit,lists[value][5],lists[value][6],lists[value][7]])
        param=0
        counta=0
        lists=[]

        
if postponingflag==0:
    #deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
    valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
    valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,5) #memo each 2-D matrix
    valuationlayers=[]
    #classified into two groups. deltaloss plus or minus and 0.
    #3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
    #use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
    semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)       
    #list sort
    valuationfirsts=functions.make_quantizedlists(semilayers,orders)
    #mix-precision part
    postponingflag=1
    print('6bit inference start')
    for i in range(len(valuationfirsts)):
        layer_index=valuationfirsts[i][0]
        block_index=valuationfirsts[i][1]
        lnum=valuationfirsts[i][2]
        cnum=valuationfirsts[i][3]
        w_bit=valuationfirsts[i][4]
        signflag=valuationfirsts[i][5]
        selectedbit=valuationfirsts[i][6]
        index=valuationfirsts[i][7]

        if lnum==100:
            break
        if lnum%2!=0 and w_bit!=0:
            param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
        if lnum%2==0 and w_bit!=0:
            param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
        #append part
        lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
        counta+=1

        if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]:
            acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = functions.KLdiv(originaloutputs,afteroutputs)
            print('preacc=',preacc)
            print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
            if acc>=preacc:
                flag=1
                postponingflag=0
                torch.save(net.state_dict(),pthname)
                preacc=acc
                params+=param
                printparams.append(params)
                printaccs.append(acc)
                printkls.append(kldiv)
                printw_bits.append(w_bit)
                printinferenceflags.append(inferenceflag)
                printlayernum.append(lnum)
                printchannels.append(counta)              
                print('success: accuracy=',acc,'kl divergence=',kldiv,'debug param=',param)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w4_bit,lists[value][5],w_bit,lists[value][7]]) 
            else: #acc<preacc->postponing
                print('postponing/reduced number of param:',param)
                if flag==0:
                    net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
                else:
                    weight = torch.load(pthname)
                    net.load_state_dict(weight)
                layers = [net.layer1,net.layer2,net.layer3,net.layer4]
                debugacc,debugloss,debugoutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
                print('debug=',debugacc)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w4_bit,lists[value][5],lists[value][6],lists[value][7]])
            param=0
            counta=0
            lists=[]



if postponingflag==0:
    #deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
    valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
    valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,6) #memo each 2-D matrix
    valuationlayers=[]
    #classified into two groups. deltaloss plus or minus and 0.
    #3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
    #use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
    semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)        
    #list sort
    valuationfirsts=functions.make_quantizedlists(semilayers,orders)
    #mix-precision part
    postponingflag=1
    print('4bit inference start')
    for i in range(len(valuationfirsts)):
        layer_index=valuationfirsts[i][0]
        block_index=valuationfirsts[i][1]
        lnum=valuationfirsts[i][2]
        cnum=valuationfirsts[i][3]
        w_bit=valuationfirsts[i][4]
        signflag=valuationfirsts[i][5]
        selectedbit=valuationfirsts[i][6]
        index=valuationfirsts[i][7]

        if lnum==100:
            break
        if lnum%2!=0 and w_bit!=0:
            param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
        if lnum%2==0 and w_bit!=0:
            param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
        #append part
        lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
        counta+=1

        if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]:
            acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = functions.KLdiv(originaloutputs,afteroutputs)
            print('preacc=',preacc)
            print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
            if acc>=preacc:
                flag=1
                postponingflag=0
                torch.save(net.state_dict(),pthname)
                preacc=acc
                params+=param
                printparams.append(params)
                printaccs.append(acc)
                printkls.append(kldiv)
                printw_bits.append(w_bit)
                printinferenceflags.append(inferenceflag)
                printlayernum.append(lnum)
                printchannels.append(counta)              
                print('success: accuracy=',acc,'kl divergence=',kldiv,'debug param=',param)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w2_bit,lists[value][5],w_bit,lists[value][7]]) 
            else: #acc<preacc->postponing
                print('postponing/reduced number of param:',param)
                if flag==0:
                    net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
                else:
                    weight = torch.load(pthname)
                    net.load_state_dict(weight)
                layers = [net.layer1,net.layer2,net.layer3,net.layer4]
                debugacc,debugloss,debugoutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
                print('debug=',debugacc)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w2_bit,lists[value][5],lists[value][6],lists[value][7]])
            param=0
            counta=0
            lists=[]


if postponingflag==0:
    #deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
    valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
    valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,7) #memo each 2-D matrix
    valuationlayers=[]
    #classified into two groups. deltaloss plus or minus and 0.
    #3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
    #use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
    semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)        
    #list sort
    valuationfirsts=functions.make_quantizedlists(semilayers,orders)
    #mix-precision part
    postponingflag=1
    print('2bit inference start')
    for i in range(len(valuationfirsts)):
        layer_index=valuationfirsts[i][0]
        block_index=valuationfirsts[i][1]
        lnum=valuationfirsts[i][2]
        cnum=valuationfirsts[i][3]
        w_bit=valuationfirsts[i][4]
        signflag=valuationfirsts[i][5]
        selectedbit=valuationfirsts[i][6]
        index=valuationfirsts[i][7]

        if lnum==100:
            break
        if lnum%2!=0 and w_bit!=0:
            param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
        if lnum%2==0 and w_bit!=0:
            param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
        #append part
        lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
        counta+=1

        if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]:
            acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = functions.KLdiv(originaloutputs,afteroutputs)
            print('preacc=',preacc)
            print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
            if acc>=preacc:
                flag=1
                postponingflag=0
                torch.save(net.state_dict(),pthname)
                preacc=acc
                params+=param
                printparams.append(params)
                printaccs.append(acc)
                printkls.append(kldiv)
                printw_bits.append(w_bit)
                printinferenceflags.append(inferenceflag)
                printlayernum.append(lnum)
                printchannels.append(counta)              
                print('success: accuracy=',acc,'kl divergence=',kldiv,'debug param=',param)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w1_bit,lists[value][5],w_bit,lists[value][7]]) 
            else: #acc<preacc->postponing
                print('postponing/reduced number of param:',param)
                if flag==0:
                    net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
                else:
                    weight = torch.load(pthname)
                    net.load_state_dict(weight)
                layers = [net.layer1,net.layer2,net.layer3,net.layer4]
                debugacc,debugloss,debugoutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
                print('debug=',debugacc)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w1_bit,lists[value][5],lists[value][6],lists[value][7]])
            param=0
            counta=0
            lists=[]


if postponingflag==0:
    #deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
    valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
    valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,8) #memo each 2-D matrix
    valuationlayers=[]
    #classified into two groups. deltaloss plus or minus and 0.
    #3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
    #use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
    semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)        
    #list sort
    valuationfirsts=functions.make_quantizedlists(semilayers,orders)
    #mix-precision part
    postponingflag=1
    print('1bit inference start')
    for i in range(len(valuationfirsts)):
        layer_index=valuationfirsts[i][0]
        block_index=valuationfirsts[i][1]
        lnum=valuationfirsts[i][2]
        cnum=valuationfirsts[i][3]
        w_bit=valuationfirsts[i][4]
        signflag=valuationfirsts[i][5]
        selectedbit=valuationfirsts[i][6]
        index=valuationfirsts[i][7]

        if lnum==100:
            break
        if lnum%2!=0 and w_bit!=0:
            param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
        if lnum%2==0 and w_bit!=0:
            param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
            layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
        #append part
        lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
        counta+=1

        if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]:
            acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = functions.KLdiv(originaloutputs,afteroutputs)
            print('preacc=',preacc)
            print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
            if acc>=preacc:
                flag=1
                postponingflag=0
                torch.save(net.state_dict(),pthname)
                preacc=acc
                params+=param
                printparams.append(params)
                printaccs.append(acc)
                printkls.append(kldiv)
                printw_bits.append(w_bit)
                printinferenceflags.append(inferenceflag)
                printlayernum.append(lnum)
                printchannels.append(counta)              
                print('success: accuracy=',acc,'kl divergence=',kldiv,'debug param=',param)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w6_bit,lists[value][5],w_bit,lists[value][7]]) 
            else: #acc<preacc->postponing
                print('postponing/reduced number of param:',param)
                if flag==0:
                    net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
                else:
                    weight = torch.load(pthname)
                    net.load_state_dict(weight)
                layers = [net.layer1,net.layer2,net.layer3,net.layer4]
                debugacc,debugloss,debugoutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
                print('debug=',debugacc)
                for value in range(len(lists)):
                    valuationlayers.append([lists[value][0],lists[value][1],lists[value][2],lists[value][3],w6_bit,lists[value][5],lists[value][6],lists[value][7]])
            param=0
            counta=0
            lists=[]


print('postponing')
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
for v in range(len(valuationlayers)):
    selectedbit=valuationlayers[v][6]
    if selectedbit==32:
        valuationnexts.append([valuationlayers[v][0],valuationlayers[v][1],valuationlayers[v][2],valuationlayers[v][3],w6_bit,0,32,valuationlayers[v][7]])
#deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit,7:2bit,8:1bit,9:pruning}
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationnexts,valuationds,5) #memo each 2-D matrix
valuationnexts=[]
#classified into two groups. deltaloss plus or minus and 0.
#3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
#use for quantized semilayer order=[index,semilayer KL divergence] index={0~31} 
semilayers,orders=functions.make_semilayers_resnet18(net2,device,originaloutputs,valuationminus,valuationplus)        
#list sort
valuationpostponings=functions.make_quantizedlists(semilayers,orders)
inferenceflag=1
weight = torch.load(pthname)
net.load_state_dict(weight)
print('final inference part')
for j in range(len(valuationpostponings)):
    layer_index=valuationpostponings[j][0]
    block_index=valuationpostponings[j][1]
    lnum=valuationpostponings[j][2]
    cnum=valuationpostponings[j][3]
    w_bit=valuationpostponings[j][4]
    signflag=valuationpostponings[j][5]
    selectedbit=valuationpostponings[j][6]

    if lnum==100:
        break
    if lnum%2!=0 and w_bit!=0:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    if lnum%2==0 and w_bit!=0:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    counta+=1

    if lnum!=valuationpostponings[j+1][2] or signflag!=valuationpostponings[j+1][5]:
        acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
        kldiv = functions.KLdiv(originaloutputs,afteroutputs)
        params+=param
        printparams.append(params)
        printaccs.append(acc)
        printkls.append(kldiv)
        printw_bits.append(w_bit)
        printinferenceflags.append(inferenceflag)
        printlayernum.append(lnum)
        printchannels.append(counta)          
        print('list append done. params=',params,'accuracy=',acc,'kl divergence=',kldiv,'layernum=',lnum,'channels:',counta)
        param=0
        counta=0


print('pth file removes')
os.remove(pthname)
end = time.time()
elapsed_time = end - start
print('elapsed_time:', elapsed_time, 'sec')
print('done')


with open("q_answer/mix/ResNet18ImageNetpq86421bit_accs_semilayer_sortnorkl_lowerdlossplusminus.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(printparams)
    writer.writerow(printaccs)
    writer.writerow(printkls)
    writer.writerow(printw_bits)
    writer.writerow(printinferenceflags)
    writer.writerow(printlayernum)
    writer.writerow(printchannels)