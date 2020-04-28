#!/usr/bin/env python

# attaches expression to the tissues for further sample generation

import os
import sys
import glob
import copy
import random
import argparse
import subprocess

import numpy as np
import pandas as pd

# ./sim_attach_exp.py --out_dir /ccb/salz8-1/avaraby/tx_noise/analysis_21042020 --base_dir_data /ccb/salz8-1/avaraby/tx_noise/data/ --base_dir_out /ccb/salz8-1/avaraby/tx_noise/data/gtex_aggs/ --num_tissues 3 --threads 30 --seed 101 > /ccb/salz8-1/avaraby/tx_noise/analysis_21042020/sim_attach_exp.log

def get_tpms(row,idx):
    tpms = row.tpms.split(";")
    res = []
    for t in tpms:
        res.append(t.split(":")[idx])
    return res

random_seed_inc = 0

def sim_attach_exp(args):
    global random_seed_inc
    random_seed_inc = args.seed
    
    base_dir_data = os.path.abspath(args.base_dir_data)+"/"
    base_dir_out = os.path.abspath(args.base_dir_out)+"/"
    out_dir = os.path.abspath(args.out_dir)+"/"

    gff3cols=["seqid","source","type","start","end","score","strand","phase","attributes"]

    gauss = pd.read_csv(base_dir_out+"res.num_tx_per_sample_loc6")
    gauss_real = gauss[(gauss["tissue_real"]>0)][["total_real",\
                                                "total_splicing",\
                                                "total_intronic",\
                                                "tissue_real",\
                                                "tissue_splicing",\
                                                "tissue_intronic",\
                                                "tpms"]].reset_index(drop=True)
    print("number of gaussians for real loci: "+str(len(gauss_real)))
    gauss_pol = gauss[gauss["tissue_intergenic"]>0][["total_intergenic",\
                                                  "tissue_intergenic",\
                                                  "tpms"]].reset_index(drop=True)
    print("number of gaussians for noise loci: "+str(len(gauss_pol)))

    for tissue_num in range(args.num_tissues):
        print("\n=================\nTissue #"+str(tissue_num)+"\n=================\n")
        pol_baseDF_sub1 = pd.read_csv(out_dir+"stage1_tissue_locs.pol_t"+str(0))

        print("starting number of intergenic transcripts: "+str(len(pol_baseDF_sub1["tid"])))
        print("starting number of intergenic genes: "+str(len(set(pol_baseDF_sub1["lid"]))))

        # first thing we need to attach tpm means and averages to each of the tissue loci
        # based on the number of transcripts, which can employ replacement if necessary
        pol_g_lid = pol_baseDF_sub1.groupby(by="lid").agg({"tid":{"tids":lambda x:list(x),"count":"count"}}).reset_index()
        pol_g_lid.columns = ["lid","tids_pol","count_pol"]

        # now attach expression values based on exact match of the number of transcripts
        num_dropped = 0
        def cond_merge_pol(g,df):
            nt = int(g["count_pol"].iloc[0])
            sub = df[df["total_intergenic"]==nt]
            if not len(sub)>=len(g):
                print(g["lid"].iloc[0],nt,len(sub),len(g))
                global num_dropped
                num_dropped+=1
                return
            sub = sub.sample(n=len(g),replace=False).reset_index(drop=True)
            g2 = pd.concat([g.reset_index(drop=True),sub],axis=1)
            assert len(g2)==len(g),"uneven length"
            return g2

        pol_g_lid = pol_g_lid.groupby('count_pol').apply(cond_merge_pol,gauss_pol).reset_index(drop=True)
        print("number of intergenic tissue loci: "+str(len(pol_g_lid)))
        print("number of intergenic loci without a match: "+str(num_dropped))

        # only keep "polymerase_num" transcripts for the tissue
        def get_n_tx(row):
            global random_seed_inc
            np.random.seed(random_seed_inc)
            random_seed_inc+=1
            tids = np.random.choice(row.tids_pol,row["tissue_intergenic"],replace=False)
            random.seed(args.seed,random_seed_inc)
            random_seed_inc+=1
            random.shuffle(tids)
            return tids

        pol_g_lid["tids_pol"] = pol_g_lid.apply(lambda row: get_n_tx(row),axis=1)

        pol_g_lid["intergenic_tpms"] = pol_g_lid.apply(lambda row: get_tpms(row,-1),axis=1)
        pol_g_lid["intergenic_tpms"] = pol_g_lid.intergenic_tpms.str.join(";")
        pol_g_lid = pol_g_lid[["lid",\
                               "tids_pol",\
                               "intergenic_tpms"]].set_index('lid').apply(lambda row: row.apply(pd.Series).stack()).reset_index().drop('level_1', 1)
        pol_g_lid.columns= ["lid","tid_intergenic","intergenic_tpms"]
        pol_g_lid.to_csv(out_dir+"stage2_tid_lid_exp.pol_t"+str(tissue_num),index=False)
        print("total number of intergenic transcripts: "+str(len(pol_g_lid)))

    for tissue_num in range(args.num_tissues):
        print("\n=================\nTissue #"+str(tissue_num)+"\n=================\n")
        real_baseDF_sub1 = pd.read_csv(out_dir+"stage1_tissue_locs.real_t"+str(tissue_num))
        splice_baseDF_sub1 = pd.read_csv(out_dir+"stage1_tissue_locs.splice_t"+str(tissue_num))
        int_baseDF_sub1 = pd.read_csv(out_dir+"stage1_tissue_locs.int_t"+str(tissue_num))

        print("starting number of real transcripts: "+str(len(real_baseDF_sub1["tid"])))
        print("starting number of real genes: "+str(len(set(real_baseDF_sub1["lid"]))))

        print("starting number of splicing transcripts: "+str(len(splice_baseDF_sub1["tid"])))
        print("starting number of splicing genes: "+str(len(set(splice_baseDF_sub1["lid"]))))

        print("starting number of intronic transcripts: "+str(len(int_baseDF_sub1["tid"])))
        print("starting number of intronic genes: "+str(len(set(int_baseDF_sub1["lid"]))))

        # and we should group them not separately, but by performing a groupby method jointly between real,splicing and intronic
        real_g_lid = real_baseDF_sub1.groupby(by="lid").agg({"tid":{"tids":lambda x:list(x),"count":"count"}}).reset_index()#[["lid","tid"]]
        real_g_lid.columns = ["lid","tids_real","count_real"]

        splice_g_lid = splice_baseDF_sub1.groupby(by="lid").agg({"tid":{"tids":lambda x:list(x),"count":"count"}}).reset_index()#[["lid","tid"]]
        splice_g_lid.columns = ["lid","tids_splice","count_splice"]

        int_g_lid = int_baseDF_sub1.groupby(by="lid").agg({"tid":{"tids":lambda x:list(x),"count":"count"}}).reset_index()#[["lid","tid"]]
        int_g_lid.columns = ["lid","tids_int","count_int"]

        all_g_lid = real_g_lid.merge(splice_g_lid,how="left",on="lid")
        all_g_lid = all_g_lid.merge(int_g_lid,how="left",on="lid")
        all_g_lid["count_splice"] = all_g_lid["count_splice"].fillna(0)
        all_g_lid["count_int"] = all_g_lid["count_int"].fillna(0)

        print("total number of real transcripts: "+str(all_g_lid["count_real"].sum()))
        print("total number of splicing transcripts: "+str(all_g_lid["count_splice"].sum()))
        print("total number of intronic transcripts: "+str(all_g_lid["count_int"].sum()))

        # now attach expression values based on exact match of the number of transcripts
        num_dropped = 0
        def cond_merge_real(g,df):
            nt_real = int(g["count_real"].iloc[0])
            nt_splicing = int(g["count_splice"].iloc[0])
            nt_intronic = int(g["count_int"].iloc[0])
            sub = df[(df["total_real"]==nt_real)&\
                     (df["total_splicing"]==nt_splicing)&\
                     (df["total_intronic"]==nt_intronic)]
            if not len(sub)>=len(g):
                print(g["lid"].iloc[0],nt_real,nt_splicing,nt_intronic)
                global num_dropped
                num_dropped+=1
                return
            sub = sub.sample(n=len(g),replace=False).reset_index(drop=True)
            g2 = pd.concat([g.reset_index(drop=True),sub],axis=1)
            assert len(g2)==len(g),"uneven length"
            return g2

        all_g_lid = all_g_lid.groupby(["count_real","count_splice","count_int"]).apply(cond_merge_real,gauss_real).reset_index(drop=True)
        print("number of real tissue loci: "+str(len(all_g_lid)))
        print("number of real loci without a match: "+str(num_dropped))
        
        # only keep "polymerase_num" transcripts for the tissue
        def get_n_tx(row):
            global random_seed_inc
            tids_real = row["tids_real"]
            new_tids_real = []
            if not row["count_real"]==0:
                np.random.seed(random_seed_inc)
                random_seed_inc+=1
                new_tids_real = np.random.choice(tids_real,row["tissue_real"],replace=False)

            tids_splice = row["tids_splice"]
            new_tids_splice = []
            if not row["count_splice"]==0:
                np.random.seed(random_seed_inc)
                random_seed_inc+=1
                new_tids_splice = np.random.choice(tids_splice,row["tissue_splicing"],replace=False)

            tids_int = row["tids_int"]
            new_tids_int = []
            if not row["count_int"]==0:
                np.random.seed(args.seed)
                new_tids_int = np.random.choice(tids_int,row["tissue_intronic"],replace=False)
            
            random.seed(args.seed,random_seed_inc)
            random_seed_inc+=1
            random.shuffle(new_tids_real)
            random.seed(args.seed,random_seed_inc)
            random_seed_inc+=1
            random.shuffle(new_tids_splice)
            random.seed(args.seed,random_seed_inc)
            random_seed_inc+=1
            random.shuffle(new_tids_int)

            return new_tids_real,new_tids_splice,new_tids_int

        all_g_lid[["tids_real","tids_splice","tids_int"]] = all_g_lid.apply(lambda row: get_n_tx(row),axis=1,result_type="expand")

        all_g_lid["real_tpms"] = all_g_lid.apply(lambda row: get_tpms(row,0),axis=1)
        all_g_lid["splicing_tpms"] = all_g_lid.apply(lambda row: get_tpms(row,1),axis=1)
        all_g_lid["intronic_tpms"] = all_g_lid.apply(lambda row: get_tpms(row,2),axis=1)
        
        all_g_lid["real_tpms"] = all_g_lid.real_tpms.str.join(";")
        all_g_lid["splicing_tpms"] = all_g_lid.splicing_tpms.str.join(";")
        all_g_lid["intronic_tpms"] = all_g_lid.intronic_tpms.str.join(";")
        
        all_g_lid = all_g_lid[["lid",\
                               "tids_real",\
                               "tids_splice",\
                               "tids_int",\
                               "real_tpms",\
                               "splicing_tpms",\
                               "intronic_tpms"]].set_index('lid').apply(lambda row: row.apply(pd.Series).stack()).reset_index().drop('level_1', 1)
        real_g_lid = all_g_lid[["lid",
                                "tids_real",
                                "real_tpms"]]
        real_g_lid.columns= ["lid","tid_real","real_tpms"]
        real_g_lid = real_g_lid[~(real_g_lid["tid_real"].isna())]
        real_g_lid.to_csv(out_dir+"stage2_tid_lid_exp.real_t"+str(tissue_num),index=False)

        splice_g_lid = all_g_lid[["lid",
                                "tids_splice",
                                "splicing_tpms"]]
        splice_g_lid.columns= ["lid","tid_splicing","splicing_tpms"]
        splice_g_lid = splice_g_lid[~(splice_g_lid["tid_splicing"].isna())]
        splice_g_lid.to_csv(out_dir+"stage2_tid_lid_exp.splice_t"+str(tissue_num),index=False)

        int_g_lid = all_g_lid[["lid",
                                "tids_int",
                                "intronic_tpms"]]
        int_g_lid.columns= ["lid","tid_intronic","intronic_tpms"]
        int_g_lid = int_g_lid[~(int_g_lid["tid_intronic"].isna())]
        int_g_lid.to_csv(out_dir+"stage2_tid_lid_exp.int_t"+str(tissue_num),index=False)

        print("total number of real transcripts: "+str(len(real_g_lid)))
        print("total number of real genes: "+str(len(set(real_g_lid["lid"]))))

        print("total number of splicing transcripts: "+str(len(splice_g_lid)))
        print("total number of splicing genes: "+str(len(set(splice_g_lid["lid"]))))

        print("total number of intronic transcripts: "+str(len(int_g_lid)))
        print("total number of intronic genes: "+str(len(set(int_g_lid["lid"]))))

    

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

    parser.set_defaults(func=sim_attach_exp)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main(sys.argv[1:])
