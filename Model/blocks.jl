#using Flux: NNlib,Chain,Conv,DepthwiseConv,MaxPool,Upsample
#include("pixDifConv.jl")


"""
CSAMblock (Channels)

Channels : number of input channels
"""
struct CSAMblock
    layers
end

CSAMblock(ch::Int) = CSAMblock(                
                Chain(
                x -> NNlib.relu.(x),
                Conv((1,1),ch =>4;pad=0),
                Conv((3,3),4=>1,NNlib.sigmoid;pad=1,bias=false)))

function (csam::CSAMblock)(x)
    y =  csam.layers(x)
    return y .* x
end


"""
CDCMblock (in_ch,out_ch)

in_ch : number of input channels
out_ch: number of output channels
"""

struct CDCMblock
    init
     b1
     b2
     b3
     b4
 end
 
CDCMblock(in_ch::Int, out_ch::Int) = CDCMblock(
    Chain( x -> NNlib.relu.(x),Conv((1,1),in_ch =>out_ch;pad=0)),
    # concat blocks
    Conv((3,3),out_ch=>out_ch ;pad=5,dilation =5,bias=false),
    Conv((3,3),out_ch=>out_ch ;pad=7,dilation =7,bias=false),
    Conv((3,3),out_ch=>out_ch ;pad=9,dilation =9,bias=false),
    Conv((3,3),out_ch=>out_ch ;pad=11,dilation =11,bias=false))
 
 
function (cdcm::CDCMblock)(x)
    x =  cdcm.init(x)
    x1 = cdcm.b1(x)
    x2 = cdcm.b2(x)
    x3 = cdcm.b3(x)
    x4 = cdcm.b4(x)
    return x1+x2+x3+x4
end

"""
mapreduce(ch)

ch = input channels
reduce the channel back to 1 channel with no padding
"""
mapReduce(ch::Int) = Conv((1,1),ch=>1,pad=0)


"""
PDCblock()

a block with Pixel Difference operations

pdc : type of effect 
inplane : input channels
output :: output chaneels
stide : default 1
"""
struct PDCblock
    stride
    convblock
    downsample
end

PDCblock(pdc,inplane::Int, outplane::Int;stride::Int=1) = PDCblock(
    stride,
    Chain(PDC((3,3),inplane =>inplane,NNlib.relu,pdc ;pad=1,bias=false,groups=inplane),
        Conv((1,1),inplane=>outplane ;pad=0,bias=false)),
    Chain(MaxPool((2,2) ; stride=(2,2)) , Conv((1,1),inplane=>outplane ;pad=0)),)


function (block::PDCblock)(x)
    if block.stride > 1
        x = block.downsample[1](x)
    end
    
    y =  block.convblock(x)
    
    if block.stride > 1
        v = block.downsample[2](x)
        return y + v
    end
    
    return y + x

end

"""
PDCblock_convert()

same as PDCblock but converts the layer into a normal convolution to reduce paramaters

pdc : type of effect 
inplane : input channels
output :: output chaneels
stide
"""
struct PDCblock_convert
    stride
    convblock
    downsample
end

convert_conv1(pdc,inplane) = (pdc == "rd" ? 
                            DepthwiseConv((5,5),inplane =>inplane,relu ;pad=2,bias=false) : 
                            DepthwiseConv((3,3),inplane =>inplane,relu ;pad=1,bias=false))

PDCblock_convert(pdc,inplane::Int, outplane::Int;stride::Int=1) = PDCblock_convert(
    stride,
    Chain( convert_conv1(pdc,inplane) , Conv((1,1),inplane=>outplane,pad=0,bias=false)),

    Chain(MaxPool((2,2) ; stride=(2,2)) , Conv((1,1),inplane=>outplane ;pad=0)),)


function (block::PDCblock_convert)(x)
    y =  block.convblock(x)
    
    if block.stride > 1
        v = downsample(x)
        return y + v
    end
    
    return y + x
end
