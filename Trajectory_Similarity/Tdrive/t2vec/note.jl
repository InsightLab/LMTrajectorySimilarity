using JSON
using Serialization
using DelimitedFiles
using Distances
using Statistics, Distributions

include("utils.jl")

datapath = "/home/jupyter-wilken.dantas@ufc.-af1ea/.conda/envs/tdrive/t2vec/data"

param = JSON.parsefile("../hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4)

println("Building spatial region with:
        cityname=$(region.name),
        minlon=$(region.minlon),
        minlat=$(region.minlat),
        maxlon=$(region.maxlon),
        maxlat=$(region.maxlat),
        xstep=$(region.xstep),
        ystep=$(region.ystep),
        minfreq=$(region.minfreq)")

paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
    println("Loaded $paramfile into region")
else
    println("Cannot find $paramfile")
end

## create querydb 
prefix = "exp1"
do_split = true
start = 400_000+20_000
num_query = 500
num_db = 50_000
querydbfile = joinpath(datapath, "$prefix-querydb.h5")
tfile = joinpath(datapath, "$prefix-trj.t")
labelfile = joinpath(datapath, "$prefix-trj.label")
vecfile = joinpath(datapath, "$prefix-trj.h5")

createQueryDB("$datapath/$cityname.h5", start, num_query, num_db,
              (x, y)->(x, y),
              (x, y)->(x, y);
              do_split=do_split,
              querydbfile=querydbfile)
createTLabel(region, querydbfile; tfile=tfile, labelfile=labelfile)

checkpoint = joinpath(datapath, "best_model.pt")
t2vec = `python t2vec.py -mode 2 -vocab_size 19693 -checkpoint $checkpoint -prefix $prefix`
println(t2vec)

cd("/home/jupyter-wilken.dantas@ufc.-af1ea/.conda/envs/tdrive/t2vec/")
run(t2vec)
cd("/home/jupyter-wilken.dantas@ufc.-af1ea/.conda/envs/tdrive/t2vec/experiment")
pwd()

println("Uiii....")
## load vectors and labels
vecs = h5open(vecfile, "r") do f
    read(f["layer3"])
end
label = readdlm(labelfile, Int)

query, db = vecs[:, 1:num_query], vecs[:, num_query+1:end]
queryLabel, dbLabel = label[1:num_query], label[num_query+1:end]
query, db = [query[:, i] for i in 1:size(query, 2)], [db[:, i] for i in 1:size(db, 2)];

# Acuracia
function acc(ranks)
    count = 0
    for i in 1:length(ranks)
        if ranks[i] == 1
            count += 1
        end
    end
    return round(count/length(ranks), digits=3)
end

# Mean Reciprocal Rainking
function mrr(ranks)
    count = 0.0
    for i in 1:length(ranks)
        count += 1.0 / ranks[i]
    end
    return round(count/length(ranks), digits=3)
end

# Intervalo de confianca para os Ranks
function cip_r(ranks)
    data = ranks

    # Create a 95% confidence interval for the population mean
    n = length(data)
    mean_val = mean(data)
    std_err = std(data) / sqrt(n)
    t_value = quantile(TDist(n - 1), 0.975)  # 95% confidence level for two-tailed test
    margin_error = t_value * std_err
    lower_bound = mean_val - margin_error
    upper_bound = mean_val + margin_error
    ic = (lower_bound, upper_bound)

    return round.(ic, digits=3)
end

# Intervalo de confianca para os Reciprocal Ranks
function cip_rr(ranks)
    data = [1 / rank for rank in ranks]

    # Create a 95% confidence interval for the population mean
    n = length(data)
    mean_val = mean(data)
    std_err = std(data) / sqrt(n)
    t_value = quantile(TDist(n - 1), 0.975)  # 95% confidence level for two-tailed test
    margin_error = t_value * std_err
    lower_bound = mean_val - margin_error
    upper_bound = mean_val + margin_error
    ic = (lower_bound, upper_bound)

    return round.(ic, digits=3)
end

# without discriminative loss
dbsizes = [10_000, 20_000, 30_000, 40_000, 50_000]
for dbsize in dbsizes
    ranks = ranksearch(query, queryLabel, db[1:dbsize], dbLabel[1:dbsize], euclidean)
    println("mean rank: $(mean(ranks)) $(cip_r(ranks)), Acc: $(acc(ranks)), MRR: $(mrr(ranks)) $(cip_rr(ranks)), with dbsize: $dbsize")
end
