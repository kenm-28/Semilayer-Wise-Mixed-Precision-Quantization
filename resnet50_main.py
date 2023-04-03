import torch
import csv
import imagenet
import functions
import time
import resnet
import os

#speed-up training and inference
torch.backends.cudnn.benchmark=True

# weight bit-width
w8_bit=8
w6_bit=6
w4_bit=4

#parameters initialized
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#list initialized
valuationds=[]
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

pthname='resnet50_semilayer_postponing.pth'

start = time.time()

#model initialized
net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
net2 = resnet.resnet50(num_classes=1000, pretrained='imagenet')
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
with open("dataset/resnet50_deltaloss.csv",encoding = "utf-8-sig") as f:
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
        count += 1
    
    for num in range(len(first)):
        lnum = int(first[num])
        cnum = int(second[num])-1
        dl8value = float(third[num])
        dl6value = float(four[num])
        dl4value = float(five[num])

        if 1<=lnum<=3:
            layer_index = 0
            block_index = 0            
        elif 4<=lnum<=6:
            layer_index = 0
            block_index = 1 
        elif 7<=lnum<=9:
            layer_index = 0
            block_index = 2
        elif 10<=lnum<=12:           
            layer_index = 1
            block_index = 0             
        elif 13<=lnum<=15:
            layer_index = 1
            block_index = 1
        elif 16<=lnum<=18:
            layer_index = 1
            block_index = 2
        elif 19<=lnum<=21:
            layer_index = 1
            block_index = 3
        elif 22<=lnum<=24:
            layer_index = 2
            block_index = 0            
        elif 25<=lnum<=27:
            layer_index = 2
            block_index = 1 
        elif 28<=lnum<=30:
            layer_index = 2
            block_index = 2
        elif 31<=lnum<=33:          
            layer_index = 2
            block_index = 3             
        elif 34<=lnum<=36:
            layer_index = 2
            block_index = 4
        elif 37<=lnum<=39:
            layer_index = 2
            block_index = 5
        elif 40<=lnum<=42:
            layer_index = 3
            block_index = 0  
        elif 43<=lnum<=45:
            layer_index = 3
            block_index = 1   
        else:
            layer_index = 3
            block_index = 2

        #normalized deltaloss part
        if lnum%3==1:
            param=layers[layer_index][block_index].conv1.weight[cnum].data.numel()
        elif lnum%3==2:
            param=layers[layer_index][block_index].conv2.weight[cnum].data.numel()
        else:
            param=layers[layer_index][block_index].conv3.weight[cnum].data.numel()
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
        valuationds.append([layer_index,block_index,lnum,cnum,dl8value,dl6value,dl4value]) #deltaloss list index=={0:layer_index,1:block_index,2:layernum,3:channelnum,4:8bit,5:6bit,6:4bit}
        valuationlayers.append([layer_index,block_index,lnum,cnum,w8_bit,0,32,num+1]) #0:layer_index,1:block_index,2:layernum,3:channelnum,4:quantized bit,5:flag,6:selected bit,7:all channels index
print('deltaloss list done.')

#8bit
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,4) #memo each 2-D matrix
valuationlayers=[]
#classified into two groups. deltaloss positive or nagative and 0.0.
#3D-matrix ex:[[[lnum,cnum,...]],[[lnum,cnum,...]]] negative index={0=layer1,...,47=layer48} positive index={48=layer1,...,95=layer48}
#use for quantized semilayer order=[index,semilayer KL divergence] index={0~95} 
semilayers,orders=functions.make_semilayers_resnet50(net2,device,originaloutputs,valuationminus,valuationplus)        
#list sort
valuationfirsts=functions.make_quantizedlists(semilayers,orders)

#parameter initialized
params=0
param=0
lists=[]
flag=0
inferenceflag=0
counta=0
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
    if lnum%3==1:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    elif lnum%3==2:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    else:
        param += layers[layer_index][block_index].conv3.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv3.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
    #append part
    lists.append([layer_index,block_index,lnum,cnum,w_bit,signflag,selectedbit,index])
    counta+=1

    if lnum!=valuationfirsts[i+1][2] or signflag!=valuationfirsts[i+1][5]: #Calculate the accuracy after semilayer quantization. 
        acc,loss,afteroutputs = functions.evaluate_acc_loss_softmax(net, device, imagenet.val_loader)
        kldiv = functions.KLdiv(originaloutputs,afteroutputs)
        print('preacc=',preacc)
        print(w_bit,'bit quantized','acc=',acc,"KL divergence=",kldiv,'layernum=',lists[-1][2],'channels:',counta)
        if acc>=preacc: #the accuracy after semilayer quantization >= the accuracy before semilayer quantization
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
                net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
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

#6bit
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,5) #memo each 2-D matrix
valuationlayers=[]
semilayers,orders=functions.make_semilayers_resnet50(net2,device,originaloutputs,valuationminus,valuationplus)       
valuationfirsts=functions.make_quantizedlists(semilayers,orders)
#mixed-precision part
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
    if lnum%3==1:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    elif lnum%3==2:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    else:
        param += layers[layer_index][block_index].conv3.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv3.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
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
                net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
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

#4bit
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index 1~3840
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationlayers,valuationds,6) #memo each 2-D matrix
valuationlayers=[]
semilayers,orders=functions.make_semilayers_resnet50(net2,device,originaloutputs,valuationminus,valuationplus)        
valuationfirsts=functions.make_quantizedlists(semilayers,orders)
#mix-precision part
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
    if lnum%3==1:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    elif lnum%3==2:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    else:
        param += layers[layer_index][block_index].conv3.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv3.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
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
                net = resnet.resnet50(num_classes=1000, pretrained='imagenet')
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

#postponing
print('postponing')
valuationlayers.sort(key = lambda x:x[7])#sort by all channels index
for v in range(len(valuationlayers)):
    selectedbit=valuationlayers[v][6]
    if selectedbit==32: #not quantized channels
        valuationnexts.append([valuationlayers[v][0],valuationlayers[v][1],valuationlayers[v][2],valuationlayers[v][3],w6_bit,0,32,valuationlayers[v][7]])
#make of semilayers with 6bit deltaloss.
valuationminus,valuationplus=functions.make_divide_minusplusmodels(valuationnexts,valuationds,5) #memo each 2-D matrix
valuationnexts=[]
semilayers,orders=functions.make_semilayers_resnet50(net2,device,originaloutputs,valuationminus,valuationplus)        
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
    if lnum%3==1:
        param += layers[layer_index][block_index].conv1.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv1.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv1.weight.data, w_bit, cnum)
    elif lnum%3==2:
        param += layers[layer_index][block_index].conv2.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv2.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv2.weight.data, w_bit, cnum)
    else:
        param += layers[layer_index][block_index].conv3.weight[cnum].data.numel()*((32-w_bit)/32-(32-selectedbit)/32)
        layers[layer_index][block_index].conv3.weight.data = functions.channel_wise_quantizationperchan(layers[layer_index][block_index].conv3.weight.data, w_bit, cnum)
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

with open("output/resnet50ImageNetq864bit_mixedprecision_accs.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(printparams)
    writer.writerow(printaccs)
    writer.writerow(printkls)
    writer.writerow(printw_bits)
    writer.writerow(printinferenceflags)
    writer.writerow(printlayernum)
    writer.writerow(printchannels)
