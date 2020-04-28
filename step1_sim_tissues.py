#!/usr/bin/env python

# Performs typing and filtering and stats aggregation of the input dataset

import os
import sys
import glob
import copy
import random
import argparse
import subprocess

import numpy as np
import pandas as pd

# ./sim_tissues.py --out_dir /ccb/salz8-1/avaraby/tx_noise/analysis_21042020 --base_dir_data /ccb/salz8-1/avaraby/tx_noise/data/ --base_dir_out /ccb/salz8-1/avaraby/tx_noise/data/gtex_aggs/ --num_tissues 3 --seed 101 > /ccb/salz8-1/avaraby/tx_noise/analysis_21042020/sim_tissues.log

def sim_tissues(args):
    base_dir_data = os.path.abspath(args.base_dir_data)+"/"
    base_dir_out = os.path.abspath(args.base_dir_out)+"/"
    out_dir = os.path.abspath(args.out_dir)+"/"
    if not os.path.exists(out_dir):
        os.makedir(out_dir)

    gff3cols=["seqid","source","type","start","end","score","strand","phase","attributes"]

    # STAGE - 1 - generating sample specific gtfs

    # load base annotations
    print(">>>loading base annotations")
    real_baseDF = pd.read_csv(base_dir_data+"real.gtf",sep="\t",names=gff3cols)
    real_baseDF = real_baseDF[(real_baseDF["type"]=="transcript") & (real_baseDF["strand"].isin(["+","-"]))].reset_index(drop=True)

    splice_baseDF = pd.read_csv(base_dir_data+"splicing.gtf",sep="\t",names=gff3cols)
    splice_baseDF = splice_baseDF[(splice_baseDF["type"]=="transcript") & (splice_baseDF["strand"].isin(["+","-"]))].reset_index(drop=True)

    int_baseDF = pd.read_csv(base_dir_data+"intronic.gtf",sep="\t",names=gff3cols)
    int_baseDF = int_baseDF[(int_baseDF["type"]=="transcript") & (int_baseDF["strand"].isin(["+","-"]))].reset_index(drop=True)

    pol_baseDF = pd.read_csv(base_dir_data+"RNApol.gtf",sep="\t",names=gff3cols)
    pol_baseDF = pol_baseDF[pol_baseDF["type"]=="transcript"].reset_index(drop=True)

    # get all loci and transcript IDs
    print(">>>getting loci IDs")
    real_baseDF["lid"] = real_baseDF["attributes"].str.split("gene_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    splice_baseDF["lid"] = splice_baseDF["attributes"].str.split("gene_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    int_baseDF["lid"] = int_baseDF["attributes"].str.split("gene_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    pol_baseDF["lid"] = pol_baseDF["attributes"].str.split("gene_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    real_locs = set(real_baseDF["lid"])
    splice_locs = set(splice_baseDF["lid"])
    int_locs = set(int_baseDF["lid"])
    pol_locs = set(pol_baseDF["lid"])
    print("starting number of real noise loci: "+str(len(real_locs)))
    print("starting number of splicing noise loci: "+str(len(splice_locs)))
    print("starting number of intronic loci: "+str(len(int_locs)))
    print("starting number of polymerase loci: "+str(len(pol_locs)))

    print("starting number of loci with real transcripts only: "+str(len(real_locs-(splice_locs.union(int_locs)))))

        # perform cleanup, by removing any loci in 
    #   1. int that are also in real
    #   2. splice that are not in real
    #   3. pol that are in real
    int_locs = int_locs - int_locs.difference(real_locs)
    assert(len(int_locs.difference(real_locs))==0),"something wrong intronic"
    int_baseDF = int_baseDF[int_baseDF["lid"].isin(int_locs)].reset_index(drop=True)

    splice_locs = splice_locs - splice_locs.difference(real_locs)
    assert(len(splice_locs) == len(real_locs.intersection(splice_locs))),"something wrong non-intronic"
    splice_baseDF = splice_baseDF[splice_baseDF["lid"].isin(splice_locs)].reset_index(drop=True)

    pol_locs = pol_locs - real_locs.intersection(pol_locs)
    assert(len(real_locs.intersection(pol_locs))==0),"something wrong polymerase"
    pol_baseDF = pol_baseDF[pol_baseDF["lid"].isin(pol_locs)].reset_index(drop=True)

    real_baseDF["tid"] = real_baseDF["attributes"].str.split("transcript_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    real_baseDF = real_baseDF[["lid","tid"]]
    splice_baseDF["tid"] = splice_baseDF["attributes"].str.split("transcript_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    splice_baseDF = splice_baseDF[["lid","tid"]]
    int_baseDF["tid"] = int_baseDF["attributes"].str.split("transcript_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    int_baseDF = int_baseDF[["lid","tid"]]
    pol_baseDF["tid"] = pol_baseDF["attributes"].str.split("transcript_id \"",expand=True,n=1)[1].str.split("\"",expand=True,n=1)[0]
    pol_baseDF = pol_baseDF[["lid","tid"]]

    print("number of real transcripts: "+str(len(real_baseDF["tid"])))
    print("number of real genes: "+str(len(set(real_baseDF["lid"]))))

    print("number of splicing transcripts: "+str(len(splice_baseDF["tid"])))
    print("number of splicing genes: "+str(len(set(splice_baseDF["lid"]))))

    print("number of intronic transcripts: "+str(len(int_baseDF["tid"])))
    print("number of intronic genes: "+str(len(set(int_baseDF["lid"]))))

    print("number of intergenic transcripts: "+str(len(pol_baseDF["tid"])))
    print("number of intergenic genes: "+str(len(set(pol_baseDF["lid"]))))

    # load the distribution of the number of real and noise loci per tissue first
    t_loc = pd.read_csv(base_dir_out+"res.num_locs_tissue")

    # now we shall remove any outliers from the data
    q25,q50,q75 = t_loc['real'].quantile([0.25,0.5,0.75])
    iqr = q75-q25
    thw = q75+1.5*iqr
    tlw = q25-1.5*iqr
    ahw = t_loc[t_loc["real"]<=thw]["real"].max()
    alw = t_loc[t_loc["real"]>=tlw]["real"].min()
    t_loc = t_loc[(t_loc['real']<=ahw)&(t_loc['real']>=alw)]

    q25,q50,q75 = t_loc['intergenic'].quantile([0.25,0.5,0.75])
    iqr = q75-q25
    thw = q75+1.5*iqr
    tlw = q25-1.5*iqr
    ahw = t_loc[t_loc["intergenic"]<=thw]["intergenic"].max()
    alw = t_loc[t_loc["intergenic"]>=tlw]["intergenic"].min()
    t_loc = t_loc[(t_loc['intergenic']<=ahw)&(t_loc['intergenic']>=alw)]

    std_real = (t_loc["real"].mean()-t_loc["real"].min())/3
    std_intergenic = (t_loc["intergenic"].mean()-t_loc["intergenic"].min())/3
    print(t_loc["real"].mean(),std_real)
    print(t_loc["intergenic"].mean(),std_intergenic)

    random_seed_inc = args.seed

    for tissue_num in range(args.num_tissues):
        print("\n=================\nTissue #"+str(tissue_num)+"\n=================\n")
        # instead of sampling from the actual values we can get it from a distribution (assuming 2 independent gaussians)
        np.random.seed(random_seed_inc)
        random_seed_inc+=1
        ctlr = int(np.random.normal(t_loc["real"].mean(),std_real,1)[0])
        np.random.seed(random_seed_inc)
        random_seed_inc+=1
        ctln = int(np.random.normal(t_loc["intergenic"].mean(),std_intergenic,1)[0])

        print("selecting "+str(ctlr)+" real loci")
        print("selecting "+str(ctln)+" intergenic loci")
        
        all_real_locs = real_locs.union(splice_locs.union(int_locs))
        print("number of all real loci: "+str(len(all_real_locs)))

        np.random.seed(random_seed_inc)
        random_seed_inc+=1
        all_real_locs_rand = np.random.choice(list(all_real_locs),ctlr, replace=False)
        np.random.seed(random_seed_inc)
        random_seed_inc+=1
        pol_locs_rand = np.random.choice(list(pol_locs),ctln, replace=False)

        real_baseDF_sub1 = real_baseDF[real_baseDF["lid"].isin(all_real_locs_rand)].reset_index(drop=True)
        splice_baseDF_sub1 = splice_baseDF[splice_baseDF["lid"].isin(all_real_locs_rand)].reset_index(drop=True)
        int_baseDF_sub1 = int_baseDF[int_baseDF["lid"].isin(all_real_locs_rand)].reset_index(drop=True)
        pol_baseDF_sub1 = pol_baseDF[pol_baseDF["lid"].isin(pol_locs_rand)].reset_index(drop=True)

        real_baseDF_sub1.to_csv(out_dir+"stage1_tissue_locs.real_t"+str(tissue_num),index=False)
        splice_baseDF_sub1.to_csv(out_dir+"stage1_tissue_locs.splice_t"+str(tissue_num),index=False)
        int_baseDF_sub1.to_csv(out_dir+"stage1_tissue_locs.int_t"+str(tissue_num),index=False)
        pol_baseDF_sub1.to_csv(out_dir+"stage1_tissue_locs.pol_t"+str(tissue_num),index=False)

        print("number of real transcripts: "+str(len(real_baseDF_sub1["tid"])))
        print("number of real genes: "+str(len(set(real_baseDF_sub1["lid"]))))

        print("number of splicing transcripts: "+str(len(splice_baseDF_sub1["tid"])))
        print("number of splicing genes: "+str(len(set(splice_baseDF_sub1["lid"]))))

        print("number of intronic transcripts: "+str(len(int_baseDF_sub1["tid"])))
        print("number of intronic genes: "+str(len(set(int_baseDF_sub1["lid"]))))

        print("number of intergenic transcripts: "+str(len(pol_baseDF_sub1["tid"])))
        print("number of intergenic genes: "+str(len(set(pol_baseDF_sub1["lid"]))))

def main(args):
    parser = argparse.ArgumentParser(description='''Help Page''')

#===========================================
#===================BUILD===================
#===========================================
    parser.add_argument('--base_dir_data',
                        required=True,
                        type=str,
                        help="path to the organized directory with base data. The results of filtering will be stored there as well")
    parser.add_argument('--base_dir_out',
                        required=True,
                        type=str,
                        help="path where the results of gtex_stats are to be stored")
    parser.add_argument('--out_dir',
                        required=True,
                        type=str,
                        help="directory to keep the results of the simulation")
    parser.add_argument("--seed",
                        required=True,
                        type=int,
                        default=101,
                        help="seed for random number generator")
    parser.add_argument("--num_tissues",
                        required=True,
                        type=int,
                        help="number of tissues to simulate")

    parser.set_defaults(func=sim_tissues)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main(sys.argv[1:])
