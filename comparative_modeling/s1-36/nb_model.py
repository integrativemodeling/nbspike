from modeller import *
from modeller.automodel import *

LOOPS = [(107, 116)]

class NanobodyModel(automodel):
    def special_patches(self, aln):
         self.rename_segments(segment_ids="A")


class NanobodyLoopModel(loopmodel):
    def special_patches(self, aln):
         self.rename_segments(segment_ids="A")

    def select_loop_atoms(self):
        residue_ranges = []
        for l in LOOPS:
            this_resrange = self.residue_range("%d:A" % l[0], "%d:A" % l[1])
            residue_ranges.append(this_resrange)
        return Selection(*residue_ranges)
