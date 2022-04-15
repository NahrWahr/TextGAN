using MLDatasets: MNIST
using Flus.Data:DataLoader
using Flux
using CUDA
using Zygote
using UnicodePlots

LrG = 2e-4
LrD = 2e-4

BatchSize = 128
NumEpochs = 300
OutputPeriod = 100
NFeatures = 28*28
LatentDim = 100
OptDscr = ADAM(LrD)
OptGen = ADAM(LrG)

TrainX, _ = MNIST.traindata(Float32)
TrainX = 2f0 * reshape(Trainx, 28, 28, 1, :) .- 1f0 |>gpu

TrainLoader = DataLoader(TrainX, batchsize = BatchSize, shuffle=true)

discriminator = Chain(Dense(NFeatures, 1024, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(256, 1, sigmoid)) |> gpu

generator = Chain(Dense(LatentDim, 256, x -> leakyrelu(x, 0.2f0)),
                    Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
                    Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
                    Dense(1024, NFeatures, tanh)) |> gpu

function TrainDscr!(discriminator, RealData, FakeData)
	ThisBatch= size(RealData)[end]

	AllData = hcat(RealData, FakeData)
	AllTarget = [ones(eltype(RealData), 1 ThisBatch) zeros(eltype(FakeData), 1, ThisBatch)] |> gpu

	ps = Flux.params(discriminator)
	loss, pullback = Zygote.pullback(ps) do
		preds = discriminator(AllData)
		loss = Flux.Losses.binarycrossentropy(preds, AllTarget)
	end

	grads = pullback(1f0)

	Flux.update!(OptDscr, Flux.params(Discriminator), grads)

	return loss
end

function TrainGen!(discriminator, generator)

	noise = rand(LatentDim, BatchSize) |> gpu

	ps = Flux.params(generator)

	loss, back = Zygote.pullback(ps) do
		preds = discriminator(generator(noise))
		loss = Flux.Losses.binarycrossentropy(preds, 1.)
	end

	grads = back(1.0f0)
	Flux.update!(OptGen, Flux.params(generator), grads)
	return loss
end

