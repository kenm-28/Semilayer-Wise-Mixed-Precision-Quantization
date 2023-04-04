import torch
import torch.nn
import random
import resnet
import imagenet
from torch import optim


def channel_wise_quantizationperchan(tensor, bit, i):
    """Quantize channel.

    Args:
        tensor (torch.Tensor): Weight(Quantize target tensor)
        bit (int): Bitwidth
        i (int): Channel number of the convolution layer depending on number of output channels

    Return:
        channels(torch.Tensor): After quantized weight
    """
    channels = 0
    channels = tensor
    channels[i][:][:][:] = quantize_wgt(tensor[i][:][:][:], bit)
    return channels

def quantize_wgt(tensor, bit):
    """Qint method. This function is used in def channel_wise_quantizationperchan.

    Args:
        tensor (torch.Tensor): Weight(Quantize target tensor)
        bit (int): Bitwidth

    Return:
        channels (torch.Tensor): After quantized weight
    """
    min_value = torch.min(tensor).item() 
    max_value = torch.max(tensor).item()
    max_scale = max_value - min_value
    #Qmax - Qmin = 7-(-8) = 15
    scale = (max_value - min_value) / (2**bit -1)
    z = round(min_value/scale)
    q_tensor = (((tensor/scale) + z).round() - z) * scale 

    return q_tensor

def evaluate_loss(net, device, data_loader):
    """Calculate loss.

    Args:
        net (-): Quantize target model
        device (torch.device): Ditwidth
        data_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training or validation 

    Return:
        loss.item() (float): Loss function of quantized target model
    """
    #net.to(device)
    #if device == 'cuda':
        #net = torch.nn.DataParallel(net)
    net.eval()
    loss = 0
    loss_sum = 0
    count = 0
    criterion = torch.nn.CrossEntropyLoss()
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
    loss = loss_sum
    loss /= count
    #net.to("cpu")
    return loss.item()

def evaluate_acc_loss_softmax(net, device, data_loader):
    """Calculate accuracy, loss, and softmaxoutputs.

    Args:
        net (-): Quantize target model
        device (torch.device): Ditwidth
        data_loader (torch.utils.data.dataloader.DataLoader): DataLoader for training or validation 

    Return:
        acc.item() (float): Accuracy after quantization
        loss.item() (float): Loss function of quantized target model
        outputs (torch.Tensor): Softmax output
    """
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
            _, y_pred = output.max(1)
            loss = criterion(output, y)
            #Calculate softmax output
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
    """Calculate KL divergence.

    Args:
        n_out (torch.Tensor): Softmax output before quantization 
        out (torch.Tensor): Softmax output after quantization

    Return:
        KL.item() (float): KL divergence derived from the probability distribution of the softmax output after and before quantization
    """
    kls = []
    for l in range(len(out)):
        for m in range(out[l].size()[0]):
            #print(n_out[l][m].size())
            kl = (n_out[l][m] * (n_out[l][m] / out[l][m]).log()).sum()
            kls.append(kl)
     #Average of KLdiv from the number of inputs
    KL = sum(kls)/len(kls)
    return KL.item()

def make_divide_minusplusmodels(paramlists,dlists,index):
    """Calculate positiveList and negativeList.

    Args:
        paramlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, quantization bitwidth, flag=0or1, \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        dlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, deltaloss for 8-bit quantization,\
            deltaloss for 6-bit quantization, deltaloss for 4-bit quantization, deltaloss for 2-bit quantization],...,[same elements]}
        index (int): Value referring to the deltaloss corresponding to the quantization bitwidth(8-bit→4, 6-bit→5, 4-bit→6, 2-bit→7) 

    Return:
        listminus (list): negativeList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(-layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        listplus (list): positiveList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
    """
    listminus=[]
    listplus=[]
    mflag=0
    pflag=1
    for i in range(len(paramlists)):
        #negative deltaloss
        if dlists[i][index]<=0:
            listminus.append([paramlists[i][0],paramlists[i][1],paramlists[i][2],paramlists[i][3],paramlists[i][4],mflag,paramlists[i][6],paramlists[i][7]])
        #positive deltaloss
        else:
            listplus.append([paramlists[i][0],paramlists[i][1],paramlists[i][2],paramlists[i][3],paramlists[i][4],pflag,paramlists[i][6],paramlists[i][7]])
        if i==len(paramlists)-1:
            print('function debug:number of total channels=',len(listminus)+len(listplus),'No.1:',len(listminus),'No.2:',len(listplus))
            break
        if paramlists[i][2]!=paramlists[i+1][2]:
            mflag-=1
            pflag+=1
    return listminus,listplus

def make_semilayers_resnet18(net,device,originaloutputs,listminus,listplus):
    """Calculate positiveList and negativeList.

    Args:
        paramlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, quantization bitwidth, flag=0or1, \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        dlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, deltaloss for 8-bit quantization,\
            deltaloss for 6-bit quantization, deltaloss for 4-bit quantization, deltaloss for 2-bit quantization],...,[same elements]}
        index (int): Value referring to the deltaloss corresponding to the quantization bitwidth(8-bit→4, 6-bit→5, 4-bit→6, 2-bit→7) 

    Return:
        listminus (list): negativeList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(-layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        listplus (list): positiveList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
    """
    uselayers=[]
    semilayers=[]
    orders=[]
    param=0
    counta=0
    index=0
    listminus.append([0,0,100,0,0,0,0,0]) #Append dummy data
    listplus.append([0,0,100,0,0,0,0,0]) #Append dummy data
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

        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Channel of odd layer quantization
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        #Channel of even layer quantization
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        #Per semilayer
        if lnum!=listminus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
            net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
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
        
        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Channel of odd layer quantization
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        #Channel of even layer quantization
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        #Per semilayer
        if lnum!=listplus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
            net = resnet.resnet18(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    return semilayers,orders 

def make_semilayers_resnet34(net,device,originaloutputs,listminus,listplus):
    """Calculate positiveList and negativeList.

    Args:
        paramlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, quantization bitwidth, flag=0or1, \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        dlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, deltaloss for 8-bit quantization,\
            deltaloss for 6-bit quantization, deltaloss for 4-bit quantization, deltaloss for 2-bit quantization],...,[same elements]}
        index (int): Value referring to the deltaloss corresponding to the quantization bitwidth(8-bit→4, 6-bit→5, 4-bit→6, 2-bit→7) 

    Return:
        listminus (list): negativeList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(-layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        listplus (list): positiveList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
    """
    uselayers=[]
    semilayers=[]
    orders=[]
    param=0
    counta=0
    index=0
    listminus.append([0,0,100,0,0,0,0,0]) #Append dummy data
    listplus.append([0,0,100,0,0,0,0,0]) #Append dummy data
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
        
        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Channel of odd layer quantization
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        #Channel of even layer quantization
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        #Per semilayer
        if lnum!=listminus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
            net = resnet.resnet34(num_classes=1000, pretrained='imagenet')
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
        
        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Channel of odd layer quantization
        if lnum%2!=0:
            layers[layer_index][block_index].conv1.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv1.weight[cnum].data.numel()*ratio
        #Channel of even layer quantization
        else:
            layers[layer_index][block_index].conv2.weight.data = channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
            param+=layers[layer_index][block_index].conv2.weight[cnum].data.numel()*ratio
        counta+=1
        #Per semilayer
        if lnum!=listplus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
            net = resnet.resnet34(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    return semilayers,orders

def make_semilayers_resnet50(net,device,originaloutputs,listminus,listplus):
    """Calculate positiveList and negativeList.

    Args:
        paramlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, quantization bitwidth, flag=0or1, \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        dlists (list): Parameters list(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, deltaloss for 8-bit quantization,\
            deltaloss for 6-bit quantization, deltaloss for 4-bit quantization, deltaloss for 2-bit quantization],...,[same elements]}
        index (int): Value referring to the deltaloss corresponding to the quantization bitwidth(8-bit→4, 6-bit→5, 4-bit→6, 2-bit→7) 

    Return:
        listminus (list): negativeList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(-layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
        listplus (list): positiveList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
    """
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
        
        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Quantization
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
        #Per semilayer
        if lnum!=listminus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
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
        
        #IF dummy data accesses, exit the loop
        if lnum==100:
            break
        #Append part
        uselayers.append([layer_index,block_index,lnum,cnum,w_bit,flag,selectedbit,channelindex])
        #Quantization 
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
        #Per semilayer
        if lnum!=listplus[i+1][2]:
            #Append part
            semilayers.append(uselayers)
            #Calculate softmax output after and before quantization
            acc,loss,afteroutputs = evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
            #Calculate KL divergence
            kldiv = KLdiv(originaloutputs,afteroutputs)
            #Calculate sensitivity(KL divergence divided by the number of semilayer parameters) 
            kldiv = kldiv/param
            print(w_bit,'bit','semilayer-No.',index,'layernumber=',lnum,'channels=',counta,'total KL divergence=',kldiv)
            #Append part
            orders.append([index,kldiv])
            #Model initialization
            net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
            layers = [net.layer1,net.layer2,net.layer3,net.layer4]
            uselayers = []
            param=0
            counta=0
            index+=1

    return semilayers,orders

def make_quantizedlists(semilayers,orders):
    """Calculate positiveList and negativeList.

    Args:
        semilayers (list): Parameters list(3-D list structure→model:[[[...]]],layer:[[...]]),channel:[...] [[[layer_index, block_index, layernumber, channelnumber, quantization bitwidth, flag=0or1, \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]]
        orders (list): Parameters list(2-D list structure): [[layernumber-1,Sensitivity],...,[same elements]}

    Return:
        valuationfirsts (list): negativeList(2-D list structure): [[layer_index, block_index, layernumber, channelnumber, flag(-layernumber), \
            selected quantization bitwidth(original model:32) ,number of all channels in convolution layers],...,[same elements]]
    """
    valuationfirsts=[]
    orders.sort(key = lambda x:x[1])
    for i in range(len(orders)):
        inum=orders[i][0]
        for j in range(len(semilayers[inum])):
            valuationfirsts.append([semilayers[inum][j][0],semilayers[inum][j][1],semilayers[inum][j][2],semilayers[inum][j][3],semilayers[inum][j][4],semilayers[inum][j][5],semilayers[inum][j][6],semilayers[inum][j][7]])
        print('debug layernum=',semilayers[inum][-1][2],'number of channels=',len(semilayers[inum]))
    print('number of valuationfirsts list=',len(valuationfirsts))
    #Append dummy data
    valuationfirsts.append([0,0,100,0,0,0,0,0])
    return valuationfirsts
