using Flux

module MyApp

	using ArgParse
	using BSON
	using Flux
	using Images:save
	
	include("Model/Model.jl")
	import .Model
	include("Utils/Utils.jl")
	import .Utils


	function parse_commandline() 
		s = ArgParseSettings()

		@add_arg_table s begin
			"--model", "-m"
				help = "choose type of model"
				default = "tiny" 
			"image"
				help = "Choose input mage"
				required = true

		end

		return parse_args(s)
	end

	function load_model(modelName)
		modelName = lowercase(modelName)
		@assert modelName in ["tiny"] "The only available model currently is tiny"
		
		#load the latest checkpoint from file
		println("loading model")
		BSON.@load "./pretrained/pidinet_9.bson" network
		println("Network loaded")
		network
		
	end
	
	#seperate the channels of the input image
	function load_image(imgPath)
		
		
		img = Utils.convertColour(imgPath)
		@assert size(img)[3] == 3 "Please input a coloured image"
		println("input loaded")
		img
	end

	function infer_img(model,image)

		res = Base.invokelatest(model, image)
		#res = model(image)
		
		res = res[5][:,:,1]

		save("./testing/results.png",res)
		print("results saved in test folder")
	end

	Base.@ccallable function julia_main()::Cint
		try

			real_main()
		catch
			Base.invokelatest(Base.display_error, Base.catch_stack())
			return 1
		end
		return 0
	end

	function real_main()
		args_p = parse_commandline()
		network = load_model(args_p["model"])

		img = load_image(args_p["image"])
		infer_img(network,img)

	end

	if abspath(PROGRAM_FILE) == @__FILE__
		real_main()
	end

end # module
