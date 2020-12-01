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

# ./sim_normalize.py --out_dir analysis_21042020 --base_dir_data data/ --base_dir_out data/gtex_aggs/ --num_tissues 3 --num_samples 10 --seed 101 --readlen 101 > analysis_21042020/sim_normalize.log

random_seed_inc = 0

def sim_normalize(args):
    global random_seed_inc
    random_seed_inc = args.seed

    base_dir_data = os.path.abspath(args.base_dir_data)+"/"
    base_dir_out = os.path.abspath(args.base_dir_out)+"/"
    out_dir = os.path.abspath(args.out_dir)+"/"

    gff3cols=["seqid","source","type","start","end","score","strand","phase","attributes"]

    readlen_stats = pd.read_csv(base_dir_data+"readlen.stats",usecols=["readlen"])

    for label in readlen_stats.columns:
        # now we shall remove any outliers from the data
        q25,q50,q75 = readlen_stats[label].quantile([0.25,0.5,0.75])
        iqr = q75-q25
        thw = q75+0.5*iqr
        tlw = q25-0.5*iqr
        ahw = readlen_stats[readlen_stats[label]<=thw][label].max()
        alw = readlen_stats[readlen_stats[label]>=tlw][label].min()
        readlen_stats = readlen_stats[(readlen_stats[label]<=ahw)&(readlen_stats[label]>=alw)].reset_index(drop=True)

    for tissue_num in range(args.num_tissues):
        print("\n=================\nTissue #"+str(tissue_num)+"\n=================\n")
        for sample_num in range(args.num_samples):
            print("++++++\n>Sample #"+str(sample_num)+"\n++++++\n")
            np.random.seed(random_seed_inc)
            random_seed_inc+=1
            total_nreads = np.random.normal(readlen_stats["readlen"].mean(),readlen_stats["readlen"].std())
            print("number of reads in sample is: "+str(total_nreads))

            real = pd.read_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            real["tid"] = real["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            real["gid"] = real["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            realt = real[real["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            reale = real[real["type"]=="exon"].reset_index(drop=True)
            reale["elen"] = reale["end"]-reale["start"]
            reale = reale[["tid","elen"]]
            reale = reale.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(realt["tid"])==set(reale["tid"]),"number of transcripts is not the same as number of groupped exons"
            reale = realt.merge(reale,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(reale),"number of tpms different from the number of transcripts"
            reale["tpm"] = tpms["tpm"]

            splicing = pd.read_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            splicing["tid"] = splicing["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            splicing["gid"] = splicing["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            splicingt = splicing[splicing["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            splicinge = splicing[splicing["type"]=="exon"].reset_index(drop=True)
            splicinge["elen"] = splicinge["end"]-splicinge["start"]
            splicinge = splicinge[["tid","elen"]]
            splicinge = splicinge.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(splicingt["tid"])==set(splicinge["tid"]),"number of transcripts is not the same as number of groupped exons"
            splicinge = splicingt.merge(splicinge,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(splicinge),"number of tpms different from the number of transcripts"
            splicinge["tpm"] = tpms["tpm"]

            intronic = pd.read_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            intronic["tid"] = intronic["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            intronic["gid"] = intronic["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            intronict = intronic[intronic["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            intronice = intronic[intronic["type"]=="exon"].reset_index(drop=True)
            intronice["elen"] = intronice["end"]-intronice["start"]
            intronice = intronice[["tid","elen"]]
            intronice = intronice.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(intronict["tid"])==set(intronice["tid"]),"number of transcripts is not the same as number of groupped exons"
            intronice = intronict.merge(intronice,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(intronice),"number of tpms different from the number of transcripts"
            intronice["tpm"] = tpms["tpm"]

            pol = pd.read_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            pol["tid"] = pol["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            pol["gid"] = pol["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            polt = pol[pol["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            pole = pol[pol["type"]=="exon"].reset_index(drop=True)
            pole["elen"] = pole["end"]-pole["start"]
            pole = pole[["tid","elen"]]
            pole = pole.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(polt["tid"])==set(pole["tid"]),"number of transcripts is not the same as number of groupped exons"
            pole = polt.merge(pole,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(pole),"number of tpms different from the number of transcripts"
            pole["tpm"] = tpms["tpm"]
            
            joined = pd.concat([reale[["tid","elen","tpm"]],splicinge[["tid","elen","tpm"]],intronice[["tid","elen","tpm"]],pole[["tid","elen","tpm"]]],axis=0).reset_index(drop=True)
            joined["theta"] = joined["elen"]*joined["tpm"]
            denom = joined["theta"].sum()
            joined["cor"] = joined["theta"]/denom
            # now that we have all these values, we 
            joined["cov"] = (joined["cor"]*total_nreads*args.readlen)/joined["elen"]
            # now we can merge the data to comply with the original ordering
            # and proceed to write it out
            realt.merge(joined[["tid","cov"]],how="left",on="tid")[["cov"]].to_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",index=False,header=False)
            splicingt.merge(joined[["tid","cov"]],how="left",on="tid")[["cov"]].to_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",index=False,header=False)
            intronict.merge(joined[["tid","cov"]],how="left",on="tid")[["cov"]].to_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",index=False,header=False)
            polt.merge(joined[["tid","cov"]],how="left",on="tid")[["cov"]].to_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",index=False,header=False)

    # now we can also compute TPMs for each sample
    for tissue_num in range(args.num_tissues):
        print("\n=================\nTissue #"+str(tissue_num)+"\n=================\n")
        for sample_num in range(args.num_samples):
            print("++++++\n>Sample #"+str(sample_num)+"\n++++++\n")
            real = pd.read_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            real["tid"] = real["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            real["gid"] = real["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            realt = real[real["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            reale = real[real["type"]=="exon"].reset_index(drop=True)
            reale["elen"] = reale["end"]-reale["start"]
            reale = reale[["tid","elen"]]
            reale = reale.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(realt["tid"])==set(reale["tid"]),"number of transcripts is not the same as number of groupped exons"
            reale = realt.merge(reale,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(reale),"number of tpms different from the number of transcripts"
            reale["tpm"] = tpms["tpm"]
            covs = pd.read_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",names=["cov"])
            assert len(covs)==len(reale),"number of covs different from the number of transcripts"
            reale["cov"] = covs["cov"]

            splicing = pd.read_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            splicing["tid"] = splicing["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            splicing["gid"] = splicing["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            splicingt = splicing[splicing["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            splicinge = splicing[splicing["type"]=="exon"].reset_index(drop=True)
            splicinge["elen"] = splicinge["end"]-splicinge["start"]
            splicinge = splicinge[["tid","elen"]]
            splicinge = splicinge.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(splicingt["tid"])==set(splicinge["tid"]),"number of transcripts is not the same as number of groupped exons"
            splicinge = splicingt.merge(splicinge,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(splicinge),"number of tpms different from the number of transcripts"
            splicinge["tpm"] = tpms["tpm"]
            covs = pd.read_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",names=["cov"])
            assert len(covs)==len(splicinge),"number of covs different from the number of transcripts"
            splicinge["cov"] = covs["cov"]

            intronic = pd.read_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            intronic["tid"] = intronic["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            intronic["gid"] = intronic["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            intronict = intronic[intronic["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            intronice = intronic[intronic["type"]=="exon"].reset_index(drop=True)
            intronice["elen"] = intronice["end"]-intronice["start"]
            intronice = intronice[["tid","elen"]]
            intronice = intronice.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(intronict["tid"])==set(intronice["tid"]),"number of transcripts is not the same as number of groupped exons"
            intronice = intronict.merge(intronice,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(intronice),"number of tpms different from the number of transcripts"
            intronice["tpm"] = tpms["tpm"]
            covs = pd.read_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",names=["cov"])
            assert len(covs)==len(intronice),"number of covs different from the number of transcripts"
            intronice["cov"] = covs["cov"]

            pol = pd.read_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".gtf",sep="\t",names=gff3cols)
            pol["tid"] = pol["attributes"].str.split("transcript_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            pol["gid"] = pol["attributes"].str.split("gene_id \"",expand=True)[1].str.split("\"",expand=True)[0]
            polt = pol[pol["type"]=="transcript"][["tid","gid"]].reset_index(drop=True) # intended for order
            pole = pol[pol["type"]=="exon"].reset_index(drop=True)
            pole["elen"] = pole["end"]-pole["start"]
            pole = pole[["tid","elen"]]
            pole = pole.groupby("tid").agg({"elen":"sum"}).reset_index()
            assert set(polt["tid"])==set(pole["tid"]),"number of transcripts is not the same as number of groupped exons"
            pole = polt.merge(pole,how="left",on="tid")
            tpms = pd.read_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".exp",names=["tpm"])
            assert len(tpms)==len(pole),"number of tpms different from the number of transcripts"
            pole["tpm"] = tpms["tpm"]
            covs = pd.read_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".cov",names=["cov"])
            assert len(covs)==len(pole),"number of covs different from the number of transcripts"
            pole["cov"] = covs["cov"]

            joined = pd.concat([reale[["tid","elen","tpm","cov"]],\
                                splicinge[["tid","elen","tpm","cov"]],\
                                intronice[["tid","elen","tpm","cov"]],\
                                pole[["tid","elen","tpm","cov"]]],axis=0).reset_index(drop=True)
            joined["new_tpm"]=(joined["cov"]*1000000)/joined["cov"].sum()
            realt.merge(joined[["tid","new_tpm"]],how="left",on="tid")[["new_tpm"]].to_csv(out_dir+"real.t"+str(tissue_num)+"_s"+str(sample_num)+".tpm",index=False,header=False)
            splicingt.merge(joined[["tid","new_tpm"]],how="left",on="tid")[["new_tpm"]].to_csv(out_dir+"splicing.t"+str(tissue_num)+"_s"+str(sample_num)+".tpm",index=False,header=False)
            intronict.merge(joined[["tid","new_tpm"]],how="left",on="tid")[["new_tpm"]].to_csv(out_dir+"intronic.t"+str(tissue_num)+"_s"+str(sample_num)+".tpm",index=False,header=False)
            polt.merge(joined[["tid","new_tpm"]],how="left",on="tid")[["new_tpm"]].to_csv(out_dir+"intergenic.t"+str(tissue_num)+"_s"+str(sample_num)+".tpm",index=False,header=False)
    

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
    parser.add_argument("--num_samples",
                        required=True,
                        type=int,
                        help="number of samples to simulate")
    parser.add_argument("--readlen",
                        required=True,
                        type=int,
                        help="read length to simulate")

    parser.set_defaults(func=sim_normalize)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main(sys.argv[1:])
