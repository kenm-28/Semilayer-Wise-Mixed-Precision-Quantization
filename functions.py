import torch
import torch.nn
import random
import resnet
import imagenet
from torch import optim

def channel_wise_quantizationperchan(tensor, bit, i):
    channels = 0
    channels = tensor
    channels[i][:][:][:] = quantize_wgt(tensor[i][:][:][:], bit)
    return channels

def quantize_wgt(tensor, bit): #layerwise no quanatize
    min_value = torch.min(tensor).item() 
    max_value = torch.max(tensor).item()
    max_scale = max_value - min_value
    scale = (max_value - min_value) / (2**bit -1)  #Qmax - Qmin = 7-(-8) = 15
    z = round(min_value/scale)
    q_tensor = (((tensor/scale) + z).round() - z) * scale 

    return q_tensor

def evaluate_loss(net, device, data_loader):
    #net.to(device)
    #if device == 'cuda':
        #net = torch.nn.DataParallel(net)
    #cudnn.benchmark = True
    net.eval()
    loss = 0
    loss_sum = 0
    count = 0
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    ys = []
    ypreds =[]
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = net(x)
            _, y_pred = net(x).max(1)
            loss = criterion(output, y)     
        ys.append(y)
        ypreds.append(y_pred)
        loss_sum += loss
        count += 1       
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
#    print(loss_sum)
    loss = loss_sum
    loss /= count
    #net.to("cpu")
    return loss.item()

def evaluate_acc_loss_softmax(net, device, data_loader):
    net.to(device)
#    if device == 'cuda':
#        net = torch.nn.DataParallel(net)
    net.eval()
    ys = []
    ypreds = []
    outputs = []
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0
    loss_sum = 0
    count = 0
    softmax = torch.nn.Softmax(dim=1)
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = net(x)
            _, y_pred = output.max(1) #use for calculating acc
            loss = criterion(output, y) #use for calculating loss
            #use for calculating output after softmax function
            softmaxoutput = softmax(output)
            outputs.append(softmaxoutput)
        ys.append(y)
        ypreds.append(y_pred)
        loss_sum += loss
        count += 1
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    loss = loss_sum
    loss /= count
#    net.to("cpu")
    return acc.item(),loss.item(),outputs

def KLdiv(n_out, out):
    kls = []
    for l in range(len(out)):
        for m in range(out[l].size()[0]):
            #print(n_out[l][m].size())
            kl = (n_out[l][m] * (n_out[l][m] / out[l][m]).log()).sum()
            kls.append(kl)
    KL = sum(kls)/len(kls) #Average of KLdiv from 50000 input
    return KL.item()

def make_divide_minusplusmodels(paramlists,dlists,index): #by valuation value(i.e. deltaloss) return type 2-D matrix
    listminus=[]
    listplus=[]
    mflag=0
    pflag=1
    for i in range(len(paramlists)):
        if dlists[i][index]<=0:
            listminus.append([paramlists[i][0],paramlists[i][1],paramlists[i][2],paramlists[i][3],paramlists[i][4],mflag,paramlists[i][6],paramlists[i][7]])
        else:
            listplus.append([paramlists[i][0],paramlists[i][1],paramlists[i][2],paramlists[i][3],paramlists[i][4],pflag,paramlists[i][6],paramlists[i][7]])
        if i==len(paramlists)-1:
            print('function debug:number of total channels=',len(listminus)+len(listplus),'No.1:',len(listminus),'No.2:',len(listplus))
            break
        if paramlists[i][2]!=paramlists[i+1][2]:
            mflag-=1
            pflag+=1
    return listminus,listplus

def make_semilayers(net,device,originaloutputs,listminus,listplus):
    uselayers=[]
    semilayers=[]
    orders=[]
    param=0
    counta=0
    index=0
    listminus.append([0,0,100,0,0,0,0,0]) #append dummy data
    listplus.append([0,0,100,0,0,0,0,0]) #append dummy data
    layers = [net.layer1,net.layer2,net.layer3,net.layer4]
    for i in range(len(listminus)):
        layer_index=listminus[i][0]
        block_index=listminus[i][1]
        lnum=listminus[i][2]
        cnum=listminus[i][3]
        w_bit=listminus[i][4]
        flag=listminus[i][5]
        selectedbit=listminus[i][6]
        channelindex=listminus[i][7]

        if w_bit==8:
            ratio=0.75
        elif w_bit==6:
            ratio=0.8125
        elif w_bit==4:
            ratio=0.875
        elif w_bit==2:
            ratio=0.9375
        else:
            ratio=0.96875
        
        if lnum==100:
            break
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        if lnum!=listminus[i+1][2]:
            semilayers.append(uselayers)
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = KLdiv(originaloutputs,afteroutputs)
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #append part
            orders.append([index,kldiv])   
            #initialized
            net = resnet_kengo.resnet34(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    for i in range(len(listplus)):
        layer_index=listplus[i][0]
        block_index=listplus[i][1]
        lnum=listplus[i][2]
        cnum=listplus[i][3]
        w_bit=listplus[i][4]
        flag=listplus[i][5]
        selectedbit=listplus[i][6]
        channelindex=listplus[i][7]

        if w_bit==8:
            ratio=0.75
        elif w_bit==6:
            ratio=0.8125
        elif w_bit==4:
            ratio=0.875
        elif w_bit==2:
            ratio=0.9375
        else:
            ratio=0.96875
        
        if lnum==100:
            break
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        if lnum!=listplus[i+1][2]:
            semilayers.append(uselayers)
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = KLdiv(originaloutputs,afteroutputs)
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #append part
            orders.append([index,kldiv])   
            #initialized
            net = resnet_kengo.resnet34(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    return semilayers,orders 

def make_semilayers_resnet50(net,device,originaloutputs,listminus,listplus):
    uselayers=[]
    semilayers=[]
    orders=[]
    param=0
    counta=0
    index=0
    listminus.append([0,0,100,0,0,0,0,0]) #append dummy data
    listplus.append([0,0,100,0,0,0,0,0]) #append dummy data
    layers = [net.layer1,net.layer2,net.layer3,net.layer4]
    for i in range(len(listminus)):
        layer_index=listminus[i][0]
        block_index=listminus[i][1]
        lnum=listminus[i][2]
        cnum=listminus[i][3]
        w_bit=listminus[i][4]
        flag=listminus[i][5]
        selectedbit=listminus[i][6]
        channelindex=listminus[i][7]
        if w_bit==8:
            ratio=0.75
        elif w_bit==6:
            ratio=0.8125
        elif w_bit==4:
            ratio=0.875
        
        if lnum==100:
            break
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        if lnum%3==1:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        elif lnum%3==2:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        else:
            layers[layer_index][block_index].conv3.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv3.weight[cnum].data.numel()*ratio
        counta+=1
        if lnum!=listminus[i+1][2]:
            semilayers.append(uselayers)
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = KLdiv(originaloutputs,afteroutputs)
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #append part
            orders.append([index,kldiv])   
            #initialized
            net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    for i in range(len(listplus)):
        layer_index=listplus[i][0]
        block_index=listplus[i][1]
        lnum=listplus[i][2]
        cnum=listplus[i][3]
        w_bit=listplus[i][4]
        flag=listplus[i][5]
        selectedbit=listplus[i][6]
        channelindex=listplus[i][7]
        if w_bit==8:
            ratio=0.75
        elif w_bit==6:
            ratio=0.8125
        elif w_bit==4:
            ratio=0.875
        
        if lnum==100:
            break
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        if lnum%3==1:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        elif lnum%3==2:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        else:
            layers[layer_index][block_index].conv3.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv3.weight[cnum].data.numel()*ratio
        counta+=1
        if lnum!=listplus[i+1][2]:
            semilayers.append(uselayers)
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            kldiv = KLdiv(originaloutputs,afteroutputs)
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #append part
            orders.append([index,kldiv])   
            #initialized
            net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    return semilayers,orders

def make_quantizedlists(semilayers,orders):
    valuationfirsts=[]
    orders.sort(key = lambda x:x[1])
    for i in range(len(orders)):
        inum=orders[i][0] #minus index={0=layer1,...,15=layer16} plus index={16=layer1,...,31=layer16}
        for j in range(len(semilayers[inum])): #2-D matrix
            valuationfirsts.append([semilayers[inum][j][0],semilayers[inum][j][1],semilayers[inum][j][2],semilayers[inum][j][3],semilayers[inum][j][4],semilayers[inum][j][5],semilayers[inum][j][6],semilayers[inum][j][7]])
        print('debug layernum=',semilayers[inum][-1][2],'number of channels=',len(semilayers[inum]))
    print('number of valuationfirsts list=',len(valuationfirsts))
    valuationfirsts.append([0,0,100,0,0,0,0,0])
    return valuationfirsts
