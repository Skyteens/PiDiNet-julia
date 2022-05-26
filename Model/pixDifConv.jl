"""
Modified convolution to allow for pixel difference convolutions based on ELBP

This code is majority taken from Flux - https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl
then modified to allow for custom weights when doing the convolution
"""


"""
Utils
"""
_paddims(x::Tuple, y::Tuple) = (x..., y[(end - (length(y) - length(x) - 1)):end]...)

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

conv_reshape_bias(c) = conv_reshape_bias(c.bias, c.stride)
conv_reshape_bias(@nospecialize(bias), _) = bias
conv_reshape_bias(bias::AbstractVector, stride) = reshape(bias, map(_->1, stride)..., :, 1)

struct SamePad end

calc_padding(lt, pad, k::NTuple{N,T}, dilation, stride) where {T,N}= expand(Val(2*N), pad)

function calc_padding(lt, ::SamePad, k::NTuple{N,T}, dilation, stride) where {N,T}
  #Ref: "A guide to convolution arithmetic for deep learning" https://arxiv.org/abs/1603.07285

  k_eff = @. k + (k - 1) * (dilation - 1)
  pad_amt = @. k_eff - 1
  return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end


"""
PDC is the convolutional layer with an extra variable effect 

effect is the type of pixel difference that is done 
Everything is pretty much as the same implementation as Conv in Flux except the name change


The output size is no different to a normal convolution, only the weight value are modified

"""

struct PDC{N,M,F,A,V,L}
    σ::F
    weight::A
    bias::V
    effect::L
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
    groups::Int
end
  
function PDC(w::AbstractArray{T,N}, b = true, σ = identity, effect = "CV";
            stride = 1, pad = 0, dilation = 1, groups = 1) where {T,N}

@assert size(w, N) % groups == 0 "Output channel dimension must be divisible by groups."
stride = expand(Val(N-2), stride)
dilation = expand(Val(N-2), dilation)
pad = calc_padding(PDC, pad, size(w)[1:N-2], dilation, stride)
bias = create_bias(w, b, size(w, N))
return PDC(σ, w, bias, effect, stride, pad, dilation, groups)
end

function PDC(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity,effect = "CV";
            init = glorot_uniform, stride = 1, pad = 0, dilation = 1, groups = 1,
            bias = true) where N
    
weight = pdcfilter(k, ch; init, groups)
PDC(weight, bias, σ, effect; stride, pad, dilation, groups)
end

function pdcfilter(filter::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer};
        init = glorot_uniform, groups = 1) where N
cin, cout = ch
@assert cin % groups == 0 "Input channel dimension must be divisible by groups."
@assert cout % groups == 0 "Output channel dimension must be divisible by groups."
init(filter..., cin÷groups, cout)
end

@functor PDC

"""
The weights for the radical pixel difference as there is indexing, there is a need to use a buffer so it is allowed for back propogation
"""
function rd_weights(weight)
  row,col,a,b = size(weight)
  buf = Buffer(zeros(5*5,a,b))
  edit_w = reshape(weight, (row*col, a,b))
  z_vec = zeros(Float32,row*col,a,b)
    
  pos = [1, 3, 5, 11, 15, 21, 23, 25]
  negs = [7, 8, 9, 12, 14, 17, 18, 19]
  nones = [2,4,6,10,13,16,20,22,24]
    
    
  buf[pos,:,:] = edit_w[ 2:end,:,:]
  buf[negs,:,:] = -edit_w[ 2:end,:,:]
  buf[nones,:,:] = z_vec[:,:,:]

  weights = convert(Array{Float32,4}, reshape(copy(buf), (5,5,a,b)))

  return weights
 
end

"""
To get the dimensions for convolution

There are 2 extra named arguments "weight" and "pad" which allows for custom weight and padding to be added
"""
pdc_dims(c::PDC, x::AbstractArray; weights::Array{Float32, 4} =c.weight,pad=c.pad) =
  NNlib.DenseConvDims(x, weights; stride = c.stride, padding = pad, dilation = c.dilation, groups = c.groups)

ChainRulesCore.@non_differentiable pdc_dims(::Any, ::Any)

"""
This is majoriry where the changes happened, used condition on effect to apply changes on weight

3 types of "cd" , "ad", "rd" then everything else is consided a normal convoltion
"""
function (c::PDC)(x::AbstractArray)
    σ = NNlib.fast_act(c.σ, x)
    
    @assert c.dilation in [(1,1), (2,2)] "dilation for cd_conv should be in 1 or 2"
    @assert size(c.weight)[1] == 3  && size(c.weight)[2] == 3 "kernel size for cd_conv should be 3x3"
    
    # difference from centered weight
    if c.effect == "cd"
        
        weights_c = convert(Array{Float32,4}, sum(c.weight,dims=(1,2)))

        cd_dims = pdc_dims(c, x ; weights=weights_c,pad=0)
        yc = NNlib.conv(x, weights_c , cd_dims)
        
        cdims = pdc_dims(c, x)
        y = NNlib.conv(x, c.weight, cdims)
        
        f_conv = y - yc
    
    # minus the clockwise adjacent
    elseif c.effect == "ad"
        row,col,a,b = size(c.weight)
        weight = reshape(c.weight, (row*col, a,b))
        
        weight_conv = weight - weight[[4, 1, 2, 7, 5, 3, 8, 9, 6],:,:]
        weight_conv = reshape(weight_conv, (row,col,a,b))
        
        cdims = pdc_dims(c, x ; weights=weight_conv)
        f_conv = NNlib.conv(x, weight_conv, cdims)
        
    elseif c.effect == "rd"
        # row,col,a,b = size(c.weight)
        # buffer = zeros(5*5,a,b)
        
        # weight = reshape(c.weight, (row*col, a,b))
        # buffer[[1, 3, 5, 11, 15, 21, 23, 25],:,:] = weight[ 2:end,:,:]
        # buffer[[7, 8, 9, 12, 14, 17, 18, 19],:,:] = -weight[ 2:end,:,:]
        # buffer[13,:,:] .= 0
        
        
        # buffer = convert(Array{Float32,4}, reshape(buffer, (5,5,a,b)))
        buffer = rd_weights(c.weight)
        padding = 2 .* c.dilation
        
        cdims = pdc_dims(c, x ; weights=buffer,pad=padding)
        f_conv = NNlib.conv(x, buffer, cdims)
        
    else
        cdims = pdc_dims(c, x)
        f_conv = NNlib.conv(x, c.weight, cdims)
    end
        
    # apply the activation and bias
  σ.(f_conv  .+ conv_reshape_bias(c))
end


"""
Visible utils - printing out the functions info
"""
_channels_in(l ::PDC) = size(l.weight, ndims(l.weight)-1) * l.groups
_channels_out(l::PDC) = size(l.weight, ndims(l.weight))

function Base.show(io::IO, l::PDC)
  print(io, "PDC(", size(l.weight)[1:ndims(l.weight)-2])
  print(io, ", ", _channels_in(l), " => ", _channels_out(l))
  print(io, ", ", l.effect)
  _print_PDC_opt(io, l)
  print(io, ")")
end

function _print_PDC_opt(io::IO, l)
  l.σ == identity || print(io, ", ", l.σ)
  all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
  all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
  all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
  if hasproperty(l, :groups)
    (l.groups == 1) || print(io, ", groups=", l.groups)
  end
  (l.bias === false) && print(io, ", bias=false")
end

