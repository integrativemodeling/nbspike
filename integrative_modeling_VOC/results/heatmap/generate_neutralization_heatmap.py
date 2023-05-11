import os
import json
import numpy as np
from collections import OrderedDict
from Bio.PDB import Chain, Model, PDBParser, PDBIO

VARIANTS = ["delta", "omicron"]
REFERENCE_PDB_FILE = {"RBD": "pdb/6m0j.pdb", "NTD": "pdb/7ly3.A_modeller.pdb", "spike": "pdb/6vyb.pdb"}
REFERENCE_PDB_CHAIN = {"RBD": ["E", "A"], "NTD": ["A", "A"], "spike": ["ABC", "ABC"]}
GROUPS = {
    1: ["S1-RBD-9", "S1-RBD-15", "S1-RBD-22", "S1-RBD-24", "S1-1"],
    2: ["S1-6"],
    3: ["S1-23", "S1-36", "S1-37", "S1-48", "S1-RBD-40"],
    4: ["S1-RBD-21", "S1-RBD-29"],
    5: ["S1-RBD-16", "S1-RBD-23"],
    7: ["S1-46"],
    8: ["S1-49"],
    9: ["S2-10"]
}

def _get_avg_affinity_ratio(variant, data):
    aff_ratio_dict = OrderedDict()
    for k,v in data.items():
        resids = set(v["wuhan_epitope"]).intersection(v[f"{variant}_epitope"])
        assert len(resids) > 0
        resids = sorted(resids)
        try:
            num = v[f"ic50_{variant}"]
        except KeyError:
            print(k, v) ; input()
        denom = 1e-20 + v["ic50_wuhan"]
        aff_ratio = -np.log10(num / denom)
        for r in resids:
            if r not in aff_ratio_dict: aff_ratio_dict[r] = []
            aff_ratio_dict[r].append(aff_ratio)
    
    avg_aff_ratio = {k: np.mean(v) for k, v in aff_ratio_dict.items()}
    return avg_aff_ratio


def _get_normalized_heatmap(data):
    delta_data = {k: v for k, v in data.items() 
                  if "delta_epitope" in v and "ic50_delta" in v}
    avg_ratio_delta = _get_avg_affinity_ratio("delta", delta_data)

    omicron_data = {k: v for k, v in data.items()
                  if "omicron_epitope" in v and "ic50_omicron" in v}
    avg_ratio_omicron = _get_avg_affinity_ratio("omicron", omicron_data)

    min_ratio = min(
        min(list(avg_ratio_delta.values())),
        min(list(avg_ratio_omicron.values()))
    )

    max_ratio = max(
        max(list(avg_ratio_delta.values())),
        max(list(avg_ratio_omicron.values()))
    )

    heatmap_delta = {k: 100 * (v-min_ratio) / (max_ratio-min_ratio)
                     for k, v in avg_ratio_delta.items()}
    
    heatmap_omicron = {k: 100 * (v-min_ratio) / (max_ratio-min_ratio)
                    for k,v in avg_ratio_omicron.items()}

    return {"delta": heatmap_delta, "omicron": heatmap_omicron}


def main(group_id, variant, data, normalized_heatmap, outdir):
    # get residues for this group and this variant
    group_variant_data = {k: v for k, v in data.items() if k in GROUPS[group_id]
                          and f"{variant}_epitope" in v and f"ic50_{variant}" in v}
    
    if not group_variant_data: return
    print(f"Creating heatmap for group {group_id}, {variant}")

    resids = set()
    for k, v in group_variant_data.items():
        resids |= set(v["wuhan_epitope"]).intersection(v[f"{variant}_epitope"])
    resids = sorted(resids)

    # get heatmap for this group and variant
    group_variant_heatmap = {r: normalized_heatmap[variant][r] for r in resids}

    # set bfactors of reference PDB file (all spike chains)
    if group_id == 8:
        domain = "NTD"
    elif group_id == 9:
        domain = "spike"
    else:
        domain = "RBD"
    model = PDBParser(QUIET=True).get_structure("x", REFERENCE_PDB_FILE[domain])[0]
    for a in model.get_atoms():
        a.set_bfactor(-1)

    ref_chain_ids = list(REFERENCE_PDB_CHAIN[domain][0])
    new_chain_ids = list(REFERENCE_PDB_CHAIN[domain][1])
    new_model = Model.Model(0)
    for (x, y) in zip(ref_chain_ids, new_chain_ids):
        new_chain = Chain.Chain(y)
        for r in model[x]:
            if r.id[1] in group_variant_heatmap:
                for a in r.get_atoms():
                    a.set_bfactor(group_variant_heatmap[r.id[1]])
            new_chain.add(r)
        new_model.add(new_chain)

    # write PDB file to disk
    out_pdb_file = os.path.join(outdir, f"{variant}_heatmap.pdb")
    io = PDBIO()
    io.set_structure(new_model)
    io.save(out_pdb_file)

    # write chimerax file to disk
    cxc_template = f"cxc_templates/{variant}_{domain}_template.cxc"
    out_cxc_file = os.path.join(outdir, f"render_{variant}.cxc")
    with open(cxc_template, "r") as of:
        s = of.read()
    
    with open(out_cxc_file, "w") as of:
        of.write(s % ",".join([str(r) for r in resids]))


if __name__ == "__main__":
    # get epitope data 
    with open("epitope_data.json", "r") as of:
        data = json.load(of)
    
     # get normalized heatmaps
    heatmap = _get_normalized_heatmap(data)

    outdir = "output/heatmap_neutralization"
    for group_id in GROUPS:
        this_outdir = os.path.join(outdir, f"group-{group_id}")
        os.system(f"mkdir -p {this_outdir}")
        for variant in VARIANTS:
            main(group_id, variant, data, heatmap, this_outdir)
            



