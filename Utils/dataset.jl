using Images
using Flux.Data: DataLoader
using FileIO:readdir
using Images:load

function convertColour(imgPath)
    img = load(imgPath)
    img = ImageCore.channelview(img)
    img = permutedims(img, [2, 3, 1])
    reshape(img, (size(img)...,1))
end

function convertGT(imgPath,threshold=0.3)
    gt = load(imgPath)
    gt = Gray.(gt)
    gt = convert(Array{Float64}, gt)

    #edit the ground truth images
    zeros = findall(gt.==0)
    ignores = findall(x -> (x>0 && x < threshold), gt)
    ones = findall(gt.>=threshold)

    gt[zeros] .= 0
    gt[ones] .=1
    gt[ignores] .=2

    gt
end

function createDataSample(file,ifTrain= true,sampleSize = 2000;root=pwd()*"/HED-BSDS")
    imgs = []
    gts = []
    
    randCreate = Random.randperm(26000)[1:sampleSize]
    lineNum = 1
	#rows = readlines(file)
    for line in eachline(file)
        if lineNum in randCreate
            imgpath, gtpath = split(line, " ")
            push!(imgs,convertColour(root *'/'*strip(imgpath)))
            push!(gts,convertGT(root *'/'*strip(gtpath ),0.3))
        end
        lineNum +=1
    end
    shuffle = ifTrain ? true : false
    DataLoader((imgs  , gts ); batchsize=1,shuffle=shuffle)
end

function createDataSmall(file,ifTrain= true;root="./ExampleData")
    imgs = []
    gts = []
    for line in eachline(file)
        imgpath, gtpath = split(line, " ")
        push!(imgs,convertColour(root *'/'*strip(imgpath)))
        push!(gts,convertGT(root *'/'*strip(gtpath ),0.3))
    end
    shuffle = ifTrain ? true : false
    DataLoader((imgs  , gts ); batchsize=1,shuffle=shuffle) 
end

function createDataLarge(file,ifTrain= true;root=pwd()*"/HED-BSDS")
    imgs = []
    gts = []
	#rows = readlines(file)
    for line in eachline(file)
        imgpath, gtpath = split(line, " ")
        push!(imgs,root *'/'*strip(imgpath))
        push!(gts,root *'/'*strip(gtpath ))
    end
    shuffle = ifTrain ? true : false
    DataLoader((imgs  , gts ); batchsize=1,shuffle=shuffle) 
end

"""
create a dataset from a folder of  jpgs
"""
# function getImgs(dir,loader)
#     trainDir = dir * "images/" * loader
#     trainPaths = readdir(trainDir, join=true)
#     trainPaths = filter(contains(r"jpg"), trainPaths) #  only get jpg files
#     imgs = load.(trainPaths)

#     new = []
#     for img in imgs
#         img = convertColour(img)
#         push!(new,img)
#     end
#     new
# end
