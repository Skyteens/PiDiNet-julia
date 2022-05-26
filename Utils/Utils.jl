module Utils

	using Flux.Data: DataLoader
	using Images:load
	using Flux.Losses:binarycrossentropy
	using Statistics:mean
	using Random

    include("dataset.jl")
    include("losses.jl")

    export createDataSample,createDataLarge, createDataSmall
			deep_loss, epoch_loss,epoch_loss_large, create_mask
	
end