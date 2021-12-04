import shutil
from modeller import *
from modeller.automodel import *

env = environ()
env.io.atom_files_directory = ["../pdb"]
env.io.hetatm = False

class MissingResiduesModel(loopmodel):
    def select_atoms(self):
        # the missing regions in the structure are 72-73 and 179-186
        # these residue ranges have been shifted to 57-58 and 164-171
        # i.e. by a offset of -16, since MODELLER starts residue numbering from 1
        return Selection(self.residue_range("57:A", "58:A"),
                         self.residue_range("164:A", "171:A"))

    def special_patches(self, aln):
        self.rename_segments(segment_ids=["A"], renumber_residues=[16])
        

a = MissingResiduesModel(env, alnfile="alignment.ali",
                         knowns="7ly3.A",
                         sequence="7ly3.A_fill",
                         assess_methods=assess.DOPE,
                         loop_assess_methods=assess.DOPE)
a.starting_model = 1
a.ending_model = 1
a.md_level = refine.fast

a.loop.starting_model = 1
a.loop.ending_model = 50
a.loop.md_level = refine.fast

a.make()

# select top comparative model according to DOPE score
models = list(filter(lambda x: x["failure"] is None, a.loop.outputs))
models = sorted(models, key=(lambda x: x["DOPE score"]))
best_model = models[0]
print("\nBest comparative model is %s, with DOPE score %.2f" % \
      (best_model["name"], best_model["DOPE score"]))
shutil.copyfile(best_model["name"], "top_scoring_model.pdb")
