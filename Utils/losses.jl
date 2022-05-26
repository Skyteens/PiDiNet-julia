"""
Create a mask as per the robust loss
"""
function create_mask(labelf, beta)
    labelf = convert(Array{Float32},labelf)
    label = round.(Int,labelf)
    mask = copy(labelf)

    num_positive = sum(x->x==1, label)
    num_negative = sum(x->x==0, label)
    
    total = num_positive + num_negative
    mask[label.==1] .= 1.0 * num_negative/total
    mask[label.==0] .= beta * num_positive/total
    mask[label.==2] .=0
    
    mask
end

"""
Binary entropy loss with mask so the weights are adjusted
"""
function bce(ŷ, y,mask=1; ϵ=gpu(fill(eps(first(ŷ)), size(ŷ)...)))
    l1 = -y.*log.(ŷ .+ ϵ)
    l2 = (1 .- y).*log.(1 .- ŷ .+ ϵ)
    l1 .- l2
    l1 .* mask  
end

"""
loss for a single output same as label
"""
function single_loss(model,img,label,mask)
    
    pred= clamp.(model(img), 0.001f0, 1.f0)
    loss = binarycrossentropy(pred[:,:,1],label;agg = identity)
    total_loss= sum(loss.*mask)
    total_loss
end

"""
loss for a list of outputs for supervision
"""

function deep_loss(model,img,label,mask)
    total_loss = 0
    outputs = model(img)
    for i = 1:5
        pred = clamp.(outputs[i], 0.001f0, 1.f0)
        loss = binarycrossentropy(pred[:,:,1],label;agg = identity)
        total_loss += sum(loss.*mask)
    end
    total_loss
end

"""
get the loss of each epoch to output
"""
function epoch_loss(model,dataloader)
    imgs,labels = dataloader.data
    loss = 0
    for i = 1:length(imgs)
        mask = create_mask(labels[i],1.1)
        loss+=deep_loss(model ,imgs[i],labels[i],mask)
    end
    loss/length(imgs)
end

function epoch_loss_large(model,dataloader)
    imgs,labels = dataloader.data

    loss = 0
    for _ = 1:500
        x = Utils.convertColour(imgs[1]) 
        y = Utils.convertGT(labels[1],0.3) 
        mask = create_mask(y,1.1)
        loss+=deep_loss(model ,x, y,mask)
    end
    loss/length(imgs)
end