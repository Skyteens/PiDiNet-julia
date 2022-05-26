module Model

    using Flux: Chain,Conv,DepthwiseConv,MaxPool,Upsample
    using Flux: @functor,ChainRulesCore,glorot_uniform,create_bias,_maybetuple_string 
    using Flux: NNlib
    using Flux.Zygote:Buffer
    
	#load the 3 files
    include("network.jl")
    include("pixDifConv.jl")
    include("blocks.jl")

	#export the functions to be used in other files
    export  PDC, PDCblock ,CDCMblock, CSAMblock,mapReduce
            PiDiNet,tiny_model,CVModel
end