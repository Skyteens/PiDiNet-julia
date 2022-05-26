using Flux

module MyApp

	using Flux
	using ArgParse
	using Printf
	using Statistics
	using BSON #: @load,@save
	using BSON: @load,@save
	
	include("Model/Model.jl")
	import .Model
	include("Utils/Utils.jl")
	import .Utils

	"""
	read the arguments from the user to get the correct model and data
	"""
	function parse_commandline()
		s = ArgParseSettings()

		@add_arg_table s begin
			"--model", "-m"
				help = "choose type of model"
				default = "tiny"
			"--data", "-d"
				help = "Choose the type of data"
				default = "small"
			"--epochs","-e"
				help = "choose number of epochs"
				arg_type = Int
				default = 2

		end

		return parse_args(s)
	end

	"""
	Get the user chosen model
	"""
	function getModel(modelName)
		modelName = lowercase(modelName)
		@assert modelName in ["tiny"] "The only available model currently is tiny"
		
		println("loading model")
		network = Model.tiny_model()

		#BSON.@load "./pidinet_4.bson" network
		println("Network loaded")
		network
	end

	"""
	Get the dataset from either large or small
	"""
	function getData(dataName)
		dataName  = lowercase(dataName)
		@assert dataName in ["small","large"] "The only available dataset are small and Large"
		
		println("creating data")

		if dataName == "small"
			train_loader = Utils.createDataSmall("./example.lst")
		elseif dataName == "large"
			train_loader = Utils.createDataSample("./train_pair.lst",true,2000;root=pwd()*"/HED-BSDS")
		end
			
		println("Data loaded")
		train_loader
	end

	"""
	The training per epoch
	"""
	function train_loop!(model,loss, ps, data, opt)
		# training_loss is declared local so it will be available for logging outside the gradient calculation.
		local training_loss
		ps = Flux.Params(ps)
		for (x,y) in data
			 
			mask =  Utils.create_mask(y[1],1.1)
	  
			gs = gradient(ps) do
				training_loss = loss(model,x[1],y[1],mask)

			return training_loss
		  end
			  
		  # Insert whatever code you want here that needs training_loss, e.g. logging.
		  # logging_callback(training_loss)
		  # Insert what ever code you want here that needs gradient.
		  # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge.
		  Flux.update!(opt, ps, gs)
		  # Here you might like to check validation set accuracy, and break out to do early stopping.
		end
	end

	"""
	The main train function to start the loop 
	"""
	function train(network,train_loader,epochs)

		ps = Flux.params(network)
		opt = ADAM(0.005, (0.9, 0.99))


		println("Begin Training")
		for i = 1:epochs
			println("Epoch $i: Training...")
			train_loop!(network,Utils.deep_loss, ps, train_loader, opt)
			@info(@sprintf("[%d]: loss: %.4f", i, Utils.epoch_loss(network,train_loader)))
			#BSON.@save "./drive/MyDrive/pidinet_$i.bson" network
		end
		BSON.@save "./edge_net.bson" network

		println("Training Finished")

	end

	"""
	The main function to get the arguments and get the arguments to start e training loop
	"""

	function main()
		args_p = parse_commandline()


		loader = getData(args_p["data"])
		
		model = getModel(args_p["model"])
		
		train(model,loader,args_p["epochs"])
	end


	Base.@ccallable function julia_main()::Cint
		try
			main()
		catch
			Base.invokelatest(Base.display_error, Base.catch_stack())
			return 1
		end
		return 0
	end

	if abspath(PROGRAM_FILE) == @__FILE__
		main()
	end

end # module