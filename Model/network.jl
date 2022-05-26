#include("blocks.jl")

"""
CARV4 is the model we will be using for PDCs - 16 layers
"""
CARV4 = ["cd","ad","rd","cv",
        "cd","ad","rd","cv",
        "cd","ad","rd","cv",
        "cd","ad","rd","cv"]

"""
getPDCblocks(convert,pdcs,inplanes)

convert ::Bool = if convertPDC or not
pdcs ::Vector  = a list of the PDC effects
inplanes :: the number of inputs in first hidden layer

This gets the stages of blocks for the main structure
"""
    
function getPDCblocks(convert,pdcs,inplanes)

    # group the stages together
    # return 4 chains
    block = convert ? PDCblock_convert : PDCblock
    # need to add convert staement
    
    
    final = []
    layers = []
    push!(layers,PDC((3,3),3=>inplanes,identity,pdcs[1] ;pad=1))
    for i = 2:16
        
        if i %4 ==1 && i < 11
            push!(layers,block(pdcs[i+1],inplanes,inplanes*2;stride =2))
            inplanes *=2
        elseif i %4 ==1
            push!(layers,block(pdcs[i+1],inplanes,inplanes;stride =2))
        else
            push!(layers,block(pdcs[i],inplanes,inplanes))
        end
        
        
        if i %4 ==0
            push!(final,Chain(layers...))
            layers=[]     
        end
    end
    
    final
    
end

"""
getSideBlocks(inplanes, dil_val , att_val)

inplanes = if convertPDC or not
dil_val ::Int  = default is nothing else a integer of number of dilations
att_val :: Bool = if there is self attention or not

This gets the blocks for the side structure
"""
function getSideBlocks(inplanes, dil_val , att_val)
    dilations = []
    attentions = []
    conv_reduces = []
    
    fuses = [1,2,4,4]

    if att_val && dil_val !== nothing
        for i = 1:4
            push!(dilations, CDCMblock(fuses[i]*inplanes,dil_val))
            push!(attentions, CSAMblock(dil_val))
            push!(conv_reduces,mapReduce(dil_val))
        end
    elseif att_val
        for i = 1:4
            push!(attentions,CSAMblock(fuses[i]*inplanes))
            push!(conv_reduces,mapReduce(fuses[i]*inplanes))
        end
    elseif (dil_val !== nothing)
        for i = 1:4
            push!(dilations,CDCMblock(fuses[i]*inplanes,dil_val))
            push!(conv_reduces,mapReduce(dil_val))
        end
    else
        for i = 1:4
            push!(conv_reduces,mapReduce(fuses[i]*inplanes))
        end
    end
    
    [dilations,attentions,conv_reduces]
end

"""
This is the main network Pixel difference network
"""
struct PiDiNet
    convBlocks
    dil_val
    att_val
    sideBlocks
    classifier
end

@functor PiDiNet

#get values
PiDiNet(inplanes::Int,pdcs;dil=nothing,att=false,convert=false) = PiDiNet(
    getPDCblocks(convert,pdcs,inplanes),
    dil,
    att,
    getSideBlocks(inplanes,dil,att),
    Conv((1,1),4=>1),)

function (net::PiDiNet)(x)

    #store the original size of images - only works if all images have the same size or batch 1
    H,W = size(x)[1:2]
    
    # first go through the main structure whilst storing the output of each stage
    x1 = net.convBlocks[1](x)
    x2 = net.convBlocks[2](x1)
    x3 = net.convBlocks[3](x2)
    x4 = net.convBlocks[4](x3)
     
    # condition for the side structure function
    if net.att_val && net.dil_val !== nothing
        xf1 = net.sideBlocks[2][1](net.sideBlocks[1][1](x1))
        xf2 = net.sideBlocks[2][2](net.sideBlocks[1][2](x2))
        xf3 = net.sideBlocks[2][3](net.sideBlocks[1][3](x3))
        xf4 = net.sideBlocks[2][4](net.sideBlocks[1][4](x4))
        x_fuses = [xf1,xf2,xf3,xf4] 
    elseif net.att_val
        xf1 = net.sideBlocks[2][1](x1)
        xf2 = net.sideBlocks[2][2](x2)
        xf3 = net.sideBlocks[2][3](x3)
        xf4 = net.sideBlocks[2][4](x4)
        x_fuses = [xf1,xf2,xf3,xf4] 
    elseif net.dil_val !== nothing
        xf1 = net.sideBlocks[1][1](x1)
        xf2 = net.sideBlocks[1][2](x2)
        xf3 = net.sideBlocks[1][3](x3)
        xf4 = net.sideBlocks[1][4](x4)
        x_fuses = [xf1,xf2,xf3,xf4] 
    else
        x_fuses = [x1,x2,x3,x4] 
    end
    
 

    #upsample the outputs back to the original size
    upsample = Upsample(:bilinear, size = (H, W))
    
    #for (i, xi) in enumerate(x_fuses)
       # e = net.sideBlocks[3][i](xi)
       # e = upsample(e)
        #push!(outputs ,e)
    #end 
    e1 = net.sideBlocks[3][1](x_fuses[1])
    e1 = upsample(e1)
	e2 = net.sideBlocks[3][2](x_fuses[2])
    e2 = upsample(e2)
	e3 = net.sideBlocks[3][3](x_fuses[3])
    e3 = upsample(e3)
	e4 = net.sideBlocks[3][4](x_fuses[4])
    e4 = upsample(e4)
    
	# merge the outputs and reduce back to one channel

	classified = net.classifier(cat(e1,e2,e3,e4,dims=3))
    #return NNlib.sigmoid.(classified )
    
    #push!(outputs,classified)
    
    # return all the sigmoid ouputs including the upsampled from each block
    return [NNlib.sigmoid.(r) for r in [e1,e2,e3,e4,classified]]
end

# x = rand(Float32,50,50,3,1)
# test = PiDiNet(20,CARV4)
# print(test)


tiny_model() = PiDiNet(20,CARV4,dil=8,att=true)

CV16 = ["cv","cv","cv","cv",
        "cv","cv","cv","cv",
        "cv","cv","cv","cv",
        "cv","cv","cv","cv"]
		
CVModel() = PiDiNet(20,CV16 )

function (net::PiDiNet)(x,k)

    #store the original size of images - only works if all images have the same size or batch 1
    H,W = size(x)[1:2]
    
    # first go through the main structure whilst storing the output of each stage
    x1 = net.convBlocks[1](x)

    upsample = Upsample(:bilinear, size = (H, W))
    
    e1 = net.sideBlocks[3][1](x1)
    e1 = upsample(e1)

	# merge the outputs and reduce back to one channel

	classified = net.classifier(e1)
    return NNlib.sigmoid.(classified )
    
end